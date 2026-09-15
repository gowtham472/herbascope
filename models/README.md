# models/

Everything in this directory except this file is generated and git-ignored. Recreate it with
`python -m scripts.run_pipeline`. The API loads the release named by `manifest.json` once at
startup; publishing a new release (and restarting the API) swaps the model without touching the UI.

| Path | Produced by | Contents |
|---|---|---|
| `manifest.json` | `ml.training.publish_release` | The active release: every artifact path with its SHA-256, plus all versions |
| `pretrained/<backbone>/` | `scripts.download_assets` | Frozen DINOv2 weights + `SOURCE.json` (name, revision) |
| `configs/<encoder version>.json` | `ml.training.extract_embeddings` | Embedding recipe: backbone, input size, pooling, dihedral views |
| `configs/<quality version>.json` | `ml.training.calibrate_quality` | Calibrated image-quality bounds |
| `configs/<decision version>.json` | `ml.training.calibrate_decision` | Calibrated decision policy |
| `classifiers/classifier.joblib` | `ml.training.train_classifier` | Logistic-regression classifier |
| `classifiers/label_encoder.json` | `ml.training.train_classifier` | Class index to name mapping |
| `classifiers/training_config.json` | `ml.training.train_classifier` | Version, embedding fingerprint, hyperparameters, selection grid |
| `classifiers/calibration.json` | `ml.training.calibrate_unknown` | Unknown-detector threshold, known boundary, risk model, calibration rates |
| `indexes/references.faiss` | `ml.training.build_index` | FAISS `IndexFlatIP` over reference embeddings |
| `indexes/reference_metadata.json` | `ml.training.build_index` | Reference records and selection provenance |
| `reports/` | `ml.evaluation.evaluate` | `evaluation.json`, per-sample `evaluation_samples.csv` |

Every artifact records the embedding fingerprint (backbone + revision + preprocessing version +
input size + pooling + views + dimension). Loading fails if any fingerprint, index version or
manifest hash disagrees.
