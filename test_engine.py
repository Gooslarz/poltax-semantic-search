import json
import torch
from search_engine import HybridTaxSearchEngine
from train_adapter import TaxDomainAdapter


def load_sample_rulings():
    with open("data/tax_benchmark.json", "r", encoding="utf-8") as f:
        return json.load(f)["rulings"]


def test_embedding_dimensions_and_l2_normalization():
    rulings = load_sample_rulings()
    engine = HybridTaxSearchEngine(rulings, model_name="intfloat/multilingual-e5-small")

    assert engine.dense_doc_matrix.shape == (8, 384), "Expected 8 documents with 384-dim embeddings"
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

    assert sim_matrix.shape == (2, 8), "Expected [2 queries, 8 rulings] similarity matrix"
    assert torch.all(sim_matrix <= 1.0001) and torch.all(sim_matrix >= -1.0001), "Cosine scores out of bounds"


def test_tax_domain_adapter_preserves_unit_norm():
    adapter = TaxDomainAdapter(embedding_dim=768)
    adapter.eval()
    dummy_input = torch.randn(4, 768)
    output = adapter(dummy_input)

    assert output.shape == (4, 768)
    norms = torch.norm(output, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
