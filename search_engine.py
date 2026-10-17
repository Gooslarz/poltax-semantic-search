import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


class HybridTaxSearchEngine:

    def __init__(self, rulings: list[dict], model_name: str = "sdadas/mmlw-retrieval-roberta-base"):
        self.rulings = rulings
        self.model_name = model_name
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

        prefixed_docs = [f"{self.passage_prefix}{r['text']}" for r in self.rulings]
        self.dense_doc_matrix = self._encode_texts(prefixed_docs)

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
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

            return F.normalize(pooled, p=2, dim=1).cpu()

    def compute_similarity_matrix(self, queries: list[str], alpha: float = 0.65) -> torch.Tensor:
        tfidf_q = self.vectorizer.transform(queries)
        lexical_sim = torch.tensor(
            sklearn_cosine(tfidf_q, self.tfidf_doc_matrix), dtype=torch.float32
        )

        prefixed_queries = [f"{self.query_prefix}{q}" for q in queries]
        dense_q = self._encode_texts(prefixed_queries)
        dense_sim = dense_q @ self.dense_doc_matrix.T

        return (alpha * dense_sim) + ((1.0 - alpha) * lexical_sim)
