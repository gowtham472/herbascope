# models/

Everything in this directory except this file is generated and git-ignored. Recreate it with
`python -m scripts.run_pipeline`. The API loads these files once at startup; replacing them
(and restarting the API) swaps the model without touching the UI.

| Path | Produced by | Contents |
|---|---|---|
| `pretrained/dinov2_vits14/` | `scripts.download_assets` | Frozen DINOv2 ViT-S/14 weights + `SOURCE.json` (name, revision) |
| `classifiers/classifier.joblib` | `ml.training.train_classifier` | Logistic-regression classifier |
| `classifiers/label_encoder.json` | `ml.training.train_classifier` | Class index to name mapping |
| `classifiers/training_config.json` | `ml.training.train_classifier` | Version, embedding fingerprint, hyperparameters, selection grid |
| `classifiers/calibration.json` | `ml.training.calibrate_unknown` | Unknown-detector threshold, known boundary, risk model, calibration rates |
| `indexes/references.faiss` | `ml.training.build_index` | FAISS `IndexFlatIP` over reference embeddings |
| `indexes/reference_metadata.json` | `ml.training.build_index` | Reference records and selection provenance |
| `configs/quality-v1.json` | `ml.training.calibrate_quality` | Calibrated image-quality bounds |
| `configs/decision-v1.json` | `ml.training.calibrate_decision` | Calibrated decision policy |
| `reports/` | `ml.evaluation.evaluate` | `evaluation.json`, per-sample `evaluation_samples.csv` |

Every artifact records the embedding fingerprint (encoder + revision + preprocessing version +
dimension). The API refuses to start inference if the fingerprints or index versions disagree.
