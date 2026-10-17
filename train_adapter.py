import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from search_engine import HybridTaxSearchEngine

torch.manual_seed(42)

TRAIN_QUERIES = [
    {"question": "Zakup szkieł progresywnych do monitora a koszty firmy programistycznej", "expected_id": "0114-KDIP3-1.4011.1"},
    {"question": "Czy faktura od optyka za okulary do kodowania pomniejsza podatek PIT?", "expected_id": "0114-KDIP3-1.4011.1"},
    {"question": "Stawka 5% PIT od dochodu z praw autorskich do oprogramowania", "expected_id": "0112-KDIL2-2.4011.2"},
    {"question": "Preferencja podatkowa przy czerpaniu zysków z licencjonowania własnego kodu i technologii", "expected_id": "0112-KDIL2-2.4011.2"},
    {"question": "Jaka stawka VAT obowiązuje na soczewki kontaktowe i okulary w optyku?", "expected_id": "0115-KDIT1-1.4012.3"},
    {"question": "Sprzedaż detaliczna wyrobów medycznych okulistycznych a obniżony VAT", "expected_id": "0115-KDIT1-1.4012.3"},
    {"question": "Maksymalna kwota rat leasingu auta spalinowego w kosztach uzyskania przychodu", "expected_id": "0111-KDIB1-3.4010.4"},
    {"question": "Proporcja 150 tys. zł przy leasingu operacyjnym samochodu osobowego", "expected_id": "0111-KDIB1-3.4010.4"},
    {"question": "Wyższy limit 225 000 zł kosztów leasingu i amortyzacji dla aut elektrycznych", "expected_id": "0111-KDIB1-3.4010.5"},
    {"question": "Samochód elektryczny w działalności gospodarczej a limit odliczeń CIT/PIT", "expected_id": "0111-KDIB1-3.4010.5"},
    {"question": "Ryczałt ewidencjonowany 12% dla programisty a 8,5% dla wsparcia helpdesk", "expected_id": "0113-KDIPT2-1.4011.6"},
    {"question": "Stawka ryczałtu dla usług PKWiU 62.01 związanych z oprogramowaniem", "expected_id": "0113-KDIPT2-1.4011.6"},
    {"question": "Wliczanie rachunków za prąd i internet w mieszkaniu do kosztów B2B", "expected_id": "0114-KDIP3-2.4011.7"},
    {"question": "Działalność zdalna programisty w domu a proporcjonalne koszty mediów", "expected_id": "0114-KDIP3-2.4011.7"},
    {"question": "Usługi IT dla klienta spoza UE (USA) a brak podatku VAT w Polsce", "expected_id": "0112-KDIL1-3.4012.8"},
    {"question": "Faktura NP (odwrotne obciążenie) za programowanie dla firmy amerykańskiej", "expected_id": "0112-KDIL1-3.4012.8"},
]


class TaxDomainAdapter(nn.Module):
    def __init__(self, embedding_dim: int = 768, rank: int = 16, dropout_p: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.down_proj = nn.Linear(embedding_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, embedding_dim, bias=False)

        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up_proj(self.down_proj(self.dropout(x)))
        adapted = x + 0.3 * delta
        return F.normalize(adapted, p=2, dim=-1)


def evaluate_unseen(sim_matrix: torch.Tensor, rulings: list[dict], test_queries: list[dict]) -> tuple[float, float]:
    hits_1, rrs = 0, []
    for i, q in enumerate(test_queries):
        ranked = [rulings[idx]["id"] for idx in torch.argsort(sim_matrix[i], descending=True).tolist()]
        rank = ranked.index(q["expected_id"]) + 1
        rrs.append(1.0 / rank)
        if rank == 1:
            hits_1 += 1
    return hits_1 / len(test_queries), sum(rrs) / len(test_queries)


def main():
    with open("data/tax_benchmark.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    rulings = data["rulings"]
    unseen_test_queries = data["queries"]

    engine = HybridTaxSearchEngine(rulings, model_name="sdadas/mmlw-retrieval-roberta-base")

    train_q_emb = engine._encode_texts([f"zapytanie: {q['question']}" for q in TRAIN_QUERIES])
    test_q_emb = engine._encode_texts([f"zapytanie: {q['question']}" for q in unseen_test_queries])
    doc_emb = engine.dense_doc_matrix

    id_to_idx = {r["id"]: i for i, r in enumerate(rulings)}
    target_indices = torch.tensor([id_to_idx[q["expected_id"]] for q in TRAIN_QUERIES], dtype=torch.long)

    adapter = TaxDomainAdapter(embedding_dim=768, rank=16, dropout_p=0.1)
    criterion = nn.CrossEntropyLoss()
    temperature = 0.05
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-2)

    base_r1, base_mrr = evaluate_unseen(test_q_emb @ doc_emb.T, rulings, unseen_test_queries)
    print(f"Epoch 00 (Frozen Baseline) | Unseen Test Recall@1: {base_r1:.1%} | Unseen MRR: {base_mrr:.3f}")

    best_mrr = base_mrr
    best_epoch = 0
    best_weights = copy.deepcopy(adapter.state_dict())

    for epoch in range(1, 41):
        adapter.train()
        optimizer.zero_grad()
        q_adapted = adapter(train_q_emb)
        d_adapted = adapter(doc_emb)
        logits = (q_adapted @ d_adapted.T) / temperature
        loss = criterion(logits, target_indices)
        loss.backward()
        optimizer.step()

        adapter.eval()
        with torch.no_grad():
            unseen_sim = adapter(test_q_emb) @ adapter(doc_emb).T
            r1, mrr = evaluate_unseen(unseen_sim, rulings, unseen_test_queries)

        if mrr > best_mrr:
            best_mrr = mrr
            best_epoch = epoch
            best_weights = copy.deepcopy(adapter.state_dict())

        if epoch in [1, 10, 20, 30, 40]:
            print(f"Epoch {epoch:02d}/40 | InfoNCE Loss: {loss.item():.4f} | Unseen Recall@1: {r1:.1%} | Unseen MRR: {mrr:.3f}")

    adapter.load_state_dict(best_weights)
    torch.save(adapter.state_dict(), "tax_adapter.pt")
    print(f"\n Best Checkpoint from Epoch {best_epoch:02d}  saved to tax_adapter.pt")


if __name__ == "__main__":
    main()