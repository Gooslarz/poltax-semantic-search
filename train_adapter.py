import copy
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from search_engine import HybridTaxSearchEngine

torch.manual_seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tax_benchmark.json"

TRAIN_QUERIES = [
    {"question": "Zakup szkieł progresywnych do monitora a koszty firmy programistycznej", "expected_id": "0114-KDIP3-1.4011.1"},
    {"question": "Czy faktura od optyka za okulary do kodowania pomniejsza podatek PIT?", "expected_id": "0114-KDIP3-1.4011.1"},
    {"question": "Stawka 5% PIT od dochodu z praw autorskich do oprogramowania", "expected_id": "0112-KDIL2-2.4011.2"},
    {"question": "Preferencja podatkowa przy czerpaniu zysków z licencjonowania własnego kodu i technologii", "expected_id": "0112-KDIL2-2.4011.2"},
    {"question": "Jaka stawka VAT obowiązuje na soczewki kontaktowe i okulary w optyku?", "expected_id": "0115-KDIT1-1.4012.3"},
    {"question": "Sprzedaż detaliczna wyrobów medycznych okulistycznych a obniżony VAT", "expected_id": "0115-KDIT1-1.4012.3"},
    {"question": "Maksymalna kwota rat leasingu auta spalinowego w kosztach uzyskania przychodu", "expected_id": "0111-KDIB1-3.4010.4"},
    {"question": "Programista B2B a limit 150 tys. zł przy leasingu operacyjnym samochodu osobowego", "expected_id": "0111-KDIB1-3.4010.4"},
    {"question": "Wyższy limit 225 000 zł kosztów leasingu i amortyzacji dla aut elektrycznych", "expected_id": "0111-KDIB1-3.4010.5"},
    {"question": "Samochód elektryczny na prąd w działalności gospodarczej a limit odliczeń CIT/PIT", "expected_id": "0111-KDIB1-3.4010.5"},
    {"question": "Ryczałt ewidencjonowany 12% dla programisty a 8,5% dla wsparcia helpdesk", "expected_id": "0113-KDIPT2-1.4011.6"},
    {"question": "Stawka ryczałtu dla usług PKWiU 62.01 związanych z oprogramowaniem", "expected_id": "0113-KDIPT2-1.4011.6"},
    {"question": "Wliczanie rachunków za prąd i internet w mieszkaniu do kosztów B2B", "expected_id": "0114-KDIP3-2.4011.7"},
    {"question": "Działalność zdalna programisty w domu a proporcjonalne koszty mediów i ogrzewania", "expected_id": "0114-KDIP3-2.4011.7"},
    {"question": "Usługi IT dla klienta spoza UE (USA) a brak podatku VAT w Polsce", "expected_id": "0112-KDIL1-3.4012.8"},
    {"question": "Faktura NP (odwrotne obciążenie) za programowanie dla firmy amerykańskiej", "expected_id": "0112-KDIL1-3.4012.8"},
]

VAL_QUERIES = [
    {"question": "Czy zakup okularów korekcyjnych przez informatyka stanowi koszt podatkowy?", "expected_id": "0114-KDIP3-1.4011.1"},
    {"question": "Opodatkowanie 5 procent dochodów z tworzenia i sprzedaży praw do algorytmów", "expected_id": "0112-KDIL2-2.4011.2"},
    {"question": "Obniżony podatek VAT 8 procent przy sprzedaży soczewek i okularów korekcyjnych", "expected_id": "0115-KDIT1-1.4012.3"},
    {"question": "Usługi programistyczne a limit 150 000 zł na leasing samochodu spalinowego", "expected_id": "0111-KDIB1-3.4010.4"},
    {"question": "Kwota graniczna odpisów za auto w pełni elektryczne (bezemisyjne) w firmie", "expected_id": "0111-KDIB1-3.4010.5"},
    {"question": "Ryczałt 8,5% czy 12% dla usług technicznych IT oraz kodowania (PKWiU 62.01)", "expected_id": "0113-KDIPT2-1.4011.6"},
    {"question": "Odliczenie części opłat za energię i internet przy pracy zdalnej z prywatnego lokalu", "expected_id": "0114-KDIP3-2.4011.7"},
    {"question": "Świadczenie usług IT dla kontrahenta z USA a miejsce opodatkowania VAT (art. 28b)", "expected_id": "0112-KDIL1-3.4012.8"},
]


class TaxDomainAdapter(nn.Module):
    def __init__(self, embedding_dim: int = 768, rank: int = 16, dropout_p: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(embedding_dim, rank, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)
        self.up_proj = nn.Linear(rank, embedding_dim, bias=False)

        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up_proj(self.dropout(self.down_proj(x)))
        adapted = x + 0.3 * delta
        return F.normalize(adapted, p=2, dim=-1)


def evaluate_split(sim_matrix: torch.Tensor, rulings: list[dict], queries: list[dict]) -> tuple[float, float]:
    hits_1, rrs = 0, []
    for i, q in enumerate(queries):
        ranked = [rulings[idx]["id"] for idx in torch.argsort(sim_matrix[i], descending=True).tolist()]
        rank = ranked.index(q["expected_id"]) + 1
        rrs.append(1.0 / rank)
        if rank == 1:
            hits_1 += 1
    return hits_1 / len(queries), sum(rrs) / len(queries)


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    rulings = data["rulings"]
    unseen_test_queries = data["queries"]

    engine = HybridTaxSearchEngine(rulings, model_name="sdadas/mmlw-retrieval-roberta-base")

    train_q_emb = engine.encode_queries([q["question"] for q in TRAIN_QUERIES])
    val_q_emb = engine.encode_queries([q["question"] for q in VAL_QUERIES])
    test_q_emb = engine.encode_queries([q["question"] for q in unseen_test_queries])
    doc_emb = engine.dense_doc_matrix       # Kept frozen! No database re-indexing needed.

    id_to_idx = {r["id"]: i for i, r in enumerate(rulings)}
    target_indices = torch.tensor([id_to_idx[q["expected_id"]] for q in TRAIN_QUERIES], dtype=torch.long)

    adapter = TaxDomainAdapter(embedding_dim=768, rank=16, dropout_p=0.1)
    criterion = nn.CrossEntropyLoss()
    temperature = 0.05
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-2)

    base_val_r1, base_val_mrr = evaluate_split(val_q_emb @ doc_emb.T, rulings, VAL_QUERIES)
    base_test_r1, base_test_mrr = evaluate_split(test_q_emb @ doc_emb.T, rulings, unseen_test_queries)
    print(f"Epoch 00 (Frozen Baseline) | Val MRR: {base_val_mrr:.3f} | Unseen Test Recall@1: {base_test_r1:.1%} (MRR: {base_test_mrr:.3f})")

    best_val_mrr = base_val_mrr
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(adapter.state_dict())

    val_target_indices = torch.tensor([id_to_idx[q["expected_id"]] for q in VAL_QUERIES], dtype=torch.long)
    for epoch in range(1, 41):
        adapter.train()
        optimizer.zero_grad()
        q_adapted = adapter(train_q_emb)
        logits = (q_adapted @ doc_emb.T) / temperature
        loss = criterion(logits, target_indices)
        loss.backward()
        optimizer.step()

        adapter.eval()
        with torch.no_grad():
            val_adapted = adapter(val_q_emb)
            val_sim = val_adapted @ doc_emb.T
            val_loss = criterion(val_sim / temperature, val_target_indices).item()
            val_r1, val_mrr = evaluate_split(val_sim, rulings, VAL_QUERIES)

        if (val_mrr > best_val_mrr) or (val_mrr == best_val_mrr and val_loss < best_val_loss):
            best_val_mrr = val_mrr
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(adapter.state_dict())

        if epoch in [1, 10, 20, 30, 40]:
            print(f"Epoch {epoch:02d}/40 | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Recall@1: {val_r1:.1%} | Val MRR: {val_mrr:.3f}")

    adapter.load_state_dict(best_weights)
    save_path = BASE_DIR / "tax_adapter.pt"
    torch.save(adapter.state_dict(), save_path)

    adapter.eval()
    with torch.no_grad():
        final_test_sim = adapter(test_q_emb) @ doc_emb.T
        final_test_r1, final_test_mrr = evaluate_split(final_test_sim, rulings, unseen_test_queries)

    print(f"\nRestored Best Validation Checkpoint (Epoch {best_epoch:02d}, Val MRR = {best_val_mrr:.3f})")
    print(f"FINAL HELD-OUT TEST EVALUATION -> Unseen Recall@1: {final_test_r1:.1%} | Unseen MRR: {final_test_mrr:.3f}")


if __name__ == "__main__":
    main()