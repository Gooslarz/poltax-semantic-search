# PolTax Semantic Search: Hybrid IR & LoRA Domain Adaptation for Polish Tax Law

A modular Information Retrieval (IR) research prototype built with **PyTorch**, **Hugging Face Transformers**, and **Scikit-learn** for semantic search over Polish individual tax interpretations (*interpretacje indywidualne KIS* across PIT, VAT, and CIT).

## Key Engineering & Research Features
* **Batched Attention-Masked Neural Pooling (`search_engine.py`):** Computes L2-normalized sentence representations in configurable mini-batches (`batch_size=32`) from transformer token embeddings (`last_hidden_state`) using `attention_mask` tensor broadcasting to eliminate `<pad>` token dilution and prevent CUDA Out-Of-Memory bottlenecks.
* **Polish & Multilingual Encoders (Hugging Face):** Benchmarks `intfloat/multilingual-e5-small` (384-dim) and the native Polish retrieval encoder `sdadas/mmlw-retrieval-roberta-base` (768-dim) with encoder-specific asymmetric prefixing (`encode_queries` vs. `encode_passages`).
* **Asymmetric Query-Side LoRA Adaptation (`train_adapter.py`):** Implements a custom PyTorch `nn.Module` (`TaxDomainAdapter`) using a zero-initialized low-rank bottleneck projection (`768 -> 16 -> 768`, reducing trainable parameters by **95.8%** from 590,592 to 24,576). Adapts query representations against a frozen document index so the corpus never requires re-indexing.
* **Leak-Free Train / Validation / Test Protocol:** Trains on 16 queries with listwise **InfoNCE (`nn.CrossEntropyLoss`, $\tau=0.05$)**, performs early-stopping checkpoint selection (`tax_adapter.pt`) on a separate 8-query **Validation Split (`VAL_QUERIES`)**, and evaluates once on 8 held-out **Unseen Test Queries** across a 12-ruling corpus containing hard-negative domain distractors (e.g., *Ulga B+R vs. IP Box*, *VAT 23% Sunglasses vs. VAT 8% Medical Lenses*).
* **Automated Functional Test Suite (`test_engine.py`):** Uses `pytest` to verify mini-batched tensor shapes, unit hypersphere L2 normalization, non-zero adapter projections, and Polish legal notation handling (`art.`, `§`, `PKWiU`).

## Benchmark Results (Unseen Test Queries over 12-Ruling Corpus with Hard Distractors)

| Method / Architecture | Recall@1 | Recall@3 | MRR |
| :--- | :--- | :--- | ---: |
| 1. Scikit-learn TF-IDF (Lexical Baseline, $\alpha=0.0$) | 62.5% | 75.0% | 0.744 |
| 2. PyTorch + `multilingual-e5-small` (Dense, $\alpha=1.0$) | 62.5% | 87.5% | 0.771 |
| 3. PyTorch + Polish `MMLW-RoBERTa` (Frozen Dense, $\alpha=1.0$) | 62.5% | 87.5% | 0.760 |
| **4. PyTorch + Polish `MMLW` + Trained LoRA Adapter (InfoNCE)** | **100.0%** | **100.0%** | **1.000** |
| 5. Hybrid: Polish `MMLW` + TF-IDF ($\alpha=0.65$) | 62.5% | 100.0% | 0.792 |
| 6. Hybrid: `multilingual-e5` + TF-IDF ($\alpha=0.65$) | 75.0% | 87.5% | 0.838 |

### Per-Question Rank X-Ray (`1` = Ranked #1 out of 12 Rulings)

| Q# | Query Category | M1 (TF-IDF) | M2 (E5) | M3 (MMLW) | **M4 (MMLW + LoRA)** | M5 (Hybrid MMLW) | M6 (Hybrid E5) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q1** | Standard Semantic (B2B Glasses vs. Monitor Distractor) | 1 | 2 | 1 | **1** | 1 | 1 |
| **Q2** | Colloquial Paraphrase (IP Box 5% vs. B+R Distractor) | 1 | 2 | 4 | **1** | 1 | 1 |
| **Q3** | Standard Semantic (Optical VAT 8% vs. 23% Distractor) | 2 | 1 | 2 | **1** | 2 | 2 |
| **Q4** | Statute / Number Trap (IT + 150k Car Lease Limit) | 4 | 6 | 3 | **1** | 3 | 5 |
| **Q5** | Colloquial Paraphrase (EV 225k Limit) | 1 | 1 | 1 | **1** | 1 | 1 |
| **Q6** | Exact Legal Classification (PKWiU 62.01) | 1 | 1 | 1 | **1** | 1 | 1 |
| **Q7** | Colloquial Paraphrase (Home Office Utilities) | 5 | 1 | 1 | **1** | 2 | 1 |
| **Q8** | Standard Semantic (US Export Art. 28b VAT) | 1 | 1 | 1 | **1** | 1 | 1 |

## Quickstart
```bash
pip install -r requirements.txt
python train_adapter.py
python evaluate.py
```

For functional unit testing:
```bash
pytest test_engine.py -v
```
