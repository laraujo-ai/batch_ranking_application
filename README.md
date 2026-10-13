# batch_ranking_application

Will hold code for our airflow based DAG that will fetch images from s3 and filter them to output a smaller subset of those images that add the most value for that model at that point in time.

## Embedding a dataset

`embed-dataset` embeds every image of a detector's training dataset (EfficientNet-B0, L2-normalised) and stores the vectors in Milvus, in a collection named after the `model_id` (e.g. `YOLOX_military_assets`; characters other than letters, digits and `_` become `_`).

```bash
uv sync
uv run embed-dataset path/to/images --model-id YOLOX_military_assets
uv run embed-dataset s3://bucket/datasets/person/ --model-id YOLOX_person_detector
```

Each row is `path` (primary key) and `embedding` (COSINE index). Re-running is idempotent: existing images are overwritten. Unreadable images are skipped and listed at the end.

Options: `--milvus-uri` (default `data/milvus.db`, i.e. Milvus Lite; also `MILVUS_URI`, e.g. `http://host:19530`), `--embedder` (any timm model), `--batch-size`, `--device`, `--recreate` (drops the model's collection first). A collection refuses vectors from a different embedder or dimension.

S3 credentials come from the standard AWS env vars/profile.

```bash
uv run pytest
```
