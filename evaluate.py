import os
import json
import pandas as pd
import torch
from search_engine import HybridTaxSearchEngine
from train_adapter import TaxDomainAdapter

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 1000)


def evaluate_predictions(sim_matrix: torch.Tensor, rulings: list[dict], queries: list[dict], method_name: str):
    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_3 = 0
    per_query_ranks = []

    for i, q in enumerate(queries):
        ranked_indices = torch.argsort(sim_matrix[i], descending=True).tolist()
        ranked_ids = [rulings[idx]["id"] for idx in ranked_indices]

        rank = ranked_ids.index(q["expected_id"]) + 1
        per_query_ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
        if rank == 1:
            hits_at_1 += 1
        if rank <= 3:
            hits_at_3 += 1

    n = len(queries)
    summary = {
        "Method / Architecture": method_name,
        "Recall@1": f"{hits_at_1 / n:.1%}",
        "Recall@3": f"{hits_at_3 / n:.1%}",
        "MRR": round(sum(reciprocal_ranks) / n, 3)
    }
    return summary, per_query_ranks


def main():
    with open("data/tax_benchmark.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    rulings = data["rulings"]
    queries = data["queries"]
    question_texts = [q["question"] for q in queries]

    engine_e5 = HybridTaxSearchEngine(rulings, model_name="intfloat/multilingual-e5-small")
    engine_mmlw = HybridTaxSearchEngine(rulings, model_name="sdadas/mmlw-retrieval-roberta-base")

    configs = [
        ("1. Scikit-learn TF-IDF (Lexical, alpha=0.0)", engine_mmlw.compute_similarity_matrix(question_texts, alpha=0.0)),
        ("2. PyTorch + multilingual-e5-small (Dense, alpha=1.0)", engine_e5.compute_similarity_matrix(question_texts, alpha=1.0)),
        ("3. PyTorch + Polish MMLW-RoBERTa (Dense, alpha=1.0)", engine_mmlw.compute_similarity_matrix(question_texts, alpha=1.0)),
    ]

    if os.path.exists("tax_adapter.pt"):
        adapter = TaxDomainAdapter(embedding_dim=768, rank=16)
        adapter.load_state_dict(torch.load("tax_adapter.pt", map_location="cpu", weights_only=True))
        adapter.eval()
        with torch.no_grad():
            q_emb = engine_mmlw._encode_texts([f"zapytanie: {q}" for q in question_texts])
            adapted_dense_sim = adapter(q_emb) @ adapter(engine_mmlw.dense_doc_matrix).T
        configs.append(("4. PyTorch + Polish MMLW + Trained LoRA Adapter (Dense)", adapted_dense_sim))

    configs.extend([
        ("5. Hybrid: Polish MMLW + TF-IDF (alpha=0.65)", engine_mmlw.compute_similarity_matrix(question_texts, alpha=0.65)),
        ("6. Hybrid: multilingual-e5 + TF-IDF (alpha=0.65)", engine_e5.compute_similarity_matrix(question_texts, alpha=0.65)),
    ])

    summaries = []
    diag_df = pd.DataFrame({"Q#": [f"Q{i+1}" for i in range(len(queries))], "Category": [q["category"] for q in queries]})

    for idx, (name, sim_mat) in enumerate(configs, 1):
        summary, ranks = evaluate_predictions(sim_mat, rulings, queries, name)
        summaries.append(summary)
        diag_df[f"M{idx}_Rank"] = ranks

    print(pd.DataFrame(summaries).to_markdown(index=False))

    print("\n====================================== PER-QUESTION RANK (1 = Perfect) ======================================")
    print(diag_df.to_markdown(index=False))


if __name__ == "__main__":
    main()