import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


class HybridTaxSearchEngine:

    def __init__(
        self,
        rulings: list[dict],
        model_name: str = "sdadas/mmlw-retrieval-roberta-base",
        batch_size: int = 32
    ):
        self.rulings = rulings
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        if "mmlw" in model_name.lower():
            self.query_prefix = "zapytanie: "
            self.passage_prefix = ""
        else:
            self.query_prefix = "query: "
            self.passage_prefix = "passage: "

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        raw_texts = [r["text"] for r in self.rulings]
        self.tfidf_doc_matrix = self.vectorizer.fit_transform(raw_texts)

        self.dense_doc_matrix = self.encode_passages(raw_texts)

    def _encode_texts(self, texts: list[str], batch_size: int | None = None) -> torch.Tensor:
        bs = batch_size or self.batch_size
        all_embeddings = []

        for start_idx in range(0, len(texts), bs):
            batch_texts = texts[start_idx : start_idx + bs]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"]

                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                word_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / word_counts
                normalized = F.normalize(pooled, p=2, dim=1).cpu()
                all_embeddings.append(normalized)

        return torch.cat(all_embeddings, dim=0)

    def encode_queries(self, queries: list[str], batch_size: int | None = None) -> torch.Tensor:
        prefixed = [f"{self.query_prefix}{q}" for q in queries]
        return self._encode_texts(prefixed, batch_size=batch_size)

    def encode_passages(self, passages: list[str], batch_size: int | None = None) -> torch.Tensor:
        prefixed = [f"{self.passage_prefix}{p}" for p in passages]
        return self._encode_texts(prefixed, batch_size=batch_size)

    def compute_similarity_matrix(self, queries: list[str], alpha: float = 0.65, adapter: nn.Module | None = None) -> torch.Tensor:

        tfidf_q = self.vectorizer.transform(queries)
        lexical_sim = torch.tensor(
            sklearn_cosine(tfidf_q, self.tfidf_doc_matrix), dtype=torch.float32
        )

        dense_q = self.encode_queries(queries)
        dense_d = self.dense_doc_matrix

        if adapter is not None:
            adapter.eval()
            with torch.no_grad():
                dense_q = adapter(dense_q)

        dense_sim = dense_q @ dense_d.T
        return (alpha * dense_sim) + ((1.0 - alpha) * lexical_sim)