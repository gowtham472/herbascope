# data/

Everything in this directory except this file is generated and git-ignored. Recreate it with
`python -m scripts.run_pipeline` (see the root README). `data/raw/` is immutable once downloaded.

| Path | Produced by | Contents |
|---|---|---|
| `raw/mikrobat/` | `scripts.download_assets` | Mikrobat repository at the pinned commit + `SOURCE.json` receipt |
| `raw/dimpsar/` | `scripts.download_assets` | DIMPSAR parquet at the pinned revision + `SOURCE.json` receipt |
| `processed/dimpsar/` | `ml.training.prepare_dataset` | Seeded DIMPSAR far-OOD subset (calibration / evaluation classes) |
| `splits/*.csv` | `ml.training.prepare_dataset` | `train`, `validation`, `test`, `heldout_known`, `ambiguity_probe`, `ood_calibration`, `ood_evaluation` |
| `metadata/` | `ml.training.prepare_dataset` | `dataset_manifest.json`, `image_metadata.csv`, `split_metadata.json` |
| `embeddings/*.npz` | `ml.training.extract_embeddings` | DINOv2 embeddings per split with the embedding fingerprint |
| `references/mikrobat/` | `ml.training.build_index` | Reference-library images served by the API |
| `demo/` | `scripts.prepare_demo_cases` | Three real evaluation images for the presentation |
| `app/` | the API at runtime | SQLite analysis history and normalised sample images |

Human-readable summaries are committed under `docs/reports/`.
