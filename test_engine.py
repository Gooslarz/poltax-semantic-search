import json
from pathlib import Path
import torch
import torch.nn.functional as F
from search_engine import HybridTaxSearchEngine
from train_adapter import TaxDomainAdapter

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tax_benchmark.json"

def load_sample_rulings():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["rulings"]


def test_embedding_dimensions_and_l2_normalization():
    rulings = load_sample_rulings()
    engine = HybridTaxSearchEngine(rulings, model_name="intfloat/multilingual-e5-small", batch_size=4)

    assert engine.dense_doc_matrix.shape == (len(rulings), 384), "Expected [N_rulings, 384] embeddings"
    norms = torch.norm(engine.dense_doc_matrix, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Vectors must be L2-normalized to 1.0"


def test_hybrid_similarity_bounds_and_polish_characters():
    rulings = load_sample_rulings()
    engine = HybridTaxSearchEngine(rulings, model_name="intfloat/multilingual-e5-small")

    queries = [
        "Art. 28b ustawy o VAT a § 1 rozporządzenia — stawka 8,5% czy 12%?",
        "Załatwienie sprawy w urzędzie skarbowym (żółć, gęślą, jaźń)"
    ]
    sim_matrix = engine.compute_similarity_matrix(queries, alpha=0.65)

    assert sim_matrix.shape == (2, len(rulings)), "Expected [2 queries, N_rulings] similarity matrix"
    assert torch.all(sim_matrix <= 1.0001) and torch.all(sim_matrix >= -1.0001), "Cosine scores out of bounds"


def test_tax_domain_adapter_non_zero_projection_and_unit_norm():
    adapter = TaxDomainAdapter(embedding_dim=768, rank=16)
    # Randomize up_proj weights so delta != 0 (avoids vacuous zero-init test)
    torch.nn.init.normal_(adapter.up_proj.weight, mean=0.0, std=0.5)
    adapter.eval()

    dummy_input = F.normalize(torch.randn(4, 768), p=2, dim=-1)
    with torch.no_grad():
        output = adapter(dummy_input)

    assert output.shape == (4, 768)
    assert not torch.allclose(output, dummy_input), "Non-zero adapter weights must transform the input vector"
    norms = torch.norm(output, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Output must remain L2-normalized to 1.0"