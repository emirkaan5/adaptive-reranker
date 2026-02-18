# adaptive-reranker


Adaptive query-level labeling pipeline for retrieval + reranking.

It builds labels that choose the best model per query by combining:
- effectiveness (`nDCG`, `MRR`)
- latency cost (BM25 vs rerankers)

The pipeline currently supports:
- BM25 candidate retrieval
- multiple rerankers (for example, `CrossEncoder`, `FlagReranker`)
- CSV export with query text + per-model metrics

## Setup

### Requirements

- Python 3.12+
- `uv` (recommended) or `pip`

### Install with uv

From the `adaptive-reranker/` directory:

```bash
uv sync
```

### Install with pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Why `transformers` is pinned

`FlagEmbedding` currently works with a compatible `transformers` range, so this project pins:

```toml
transformers>=4.38,<4.45
```

Keep this constraint unless you verify your `FlagEmbedding` version with newer `transformers`.

## Usage

Minimal example:

```python
import torch
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker
from adaptive_reranker.labeling.pipeline import Labeler

DATASETS = {
    "NFCorpus": "beir/nfcorpus/test",
}

device = "cuda" if torch.cuda.is_available() else "cpu"

ce_l6 = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
bge_reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=(device == "cuda"),
)

labeler = Labeler(
    datasets=DATASETS,
    rerankers={
        "ce_l6": ce_l6,
        "bge_reranker": bge_reranker,
    },
    device=device,
    top_k_retrieve=50,
    lambda_cost=0.05,
    eval_at_k=10,
)

# Runs all datasets in DATASETS, saves CSVs, and returns DataFrames.
dfs = labeler.run(output_dir="labels")

# Example access:
# df_nfcorpus = dfs["beir/nfcorpus/test"]
```

## Output

For each dataset, the pipeline writes a CSV:

```text
labels/labels_<dataset_id_with_slashes_replaced>.csv
```

Example:

```text
labels/labels_beir_nfcorpus_test.csv
```

The CSV includes:
- `qid`, `query`
- selected class and best model (`class`, `best_model`, `best_utility`)
- per-model metrics as prefixed columns, such as:
  - `bm25_ndcg`, `bm25_mrr`, `bm25_latency`
  - `ce_l6_ndcg`, `ce_l6_utility`, ...

## Run the included example

```bash
uv run tests/generate_example.py
```

This should produce one CSV per dataset in the output directory.