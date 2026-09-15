# HerbaScope X — Architecture Decision Record

Every significant design choice, why it was made, and what was rejected. Numbers quoted here
come from the generated reports in [`docs/reports/`](reports/); rerunning the pipeline regenerates
those reports.

| ADR | Decision |
|---|---|
| [001](#adr-001-monorepo-layout) | Monorepo: `apps/web`, `services/api`, `ml`, `scripts` |
| [002](#adr-002-dataset-mikrobat-primary-dimpsar-as-far-ood) | Mikrobat primary, DIMPSAR as far-OOD negatives |
| [003](#adr-003-duplicate-and-leakage-control) | SHA-256 + pHash near-duplicate groups, bounded by a cross-species null |
| [004](#adr-004-held-out-known-material-and-the-ambiguity-probe) | Held-out fragment types + ambiguity probe |
| [005](#adr-005-preprocessing) | Grayscale, pad-to-square, no crop, versioned |
| [006](#adr-006-frozen-dinov2-encoder) | Frozen DINOv2 ViT-S/14, pinned, self-describing weights |
| [007](#adr-007-logistic-regression-classifier) | Logistic regression, C chosen by validation log-loss |
| [008](#adr-008-reference-library-and-faiss) | K-means-medoid reference library, exact FAISS search |
| [009](#adr-009-reproducible-artifacts-and-what-git-tracks) | Pinned sources; generated artifacts ignored, reports committed |
| [010](#adr-010-unknown-detection-and-calibration) | Reference-distance unknown detector with objective-based threshold |
| [011](#adr-011-image-quality) | Percentile-calibrated quality bounds |
| [012](#adr-012-evidence-engine) | Structural agreement; strength is descriptive only |
| [013](#adr-013-deterministic-decision-policy) | Ordered rules, calibrated policy file |
| [014](#adr-014-one-screening-pipeline-for-api-evaluation-and-smoke-test) | One pipeline class for API, evaluation and smoke test |
| [015](#adr-015-fastapi-service) | Single FastAPI service, SQLite history, degraded mode |
| [016](#adr-016-nextjs-frontend) | Next.js 16 App Router, client-side fetching, offline fonts |
| [017](#adr-017-no-llm) | No LLM; deterministic explanation templates |
| [018](#adr-018-testing-strategy) | Synthetic-pipeline tests + real-artifact integration tests |
| [019](#adr-019-docker) | Docker mounts artifacts; CPU torch; standalone Next |

---

## ADR-001: Monorepo layout

**Decision.** One repository with `apps/web` (Next.js), `services/api` (FastAPI), `ml/` (shared
Python library and training/evaluation modules), `scripts/` (orchestration), `tests/`, `docs/`,
`docker/`. The Next.js scaffold was moved from the root into `apps/web` with `git mv`, so its
history is preserved.

**Why a monorepo suits HerbaScope X best**

1. **One contract, one commit.** The frontend types (`apps/web/types/index.ts`), the API schema
   (`services/api/app/schemas/analysis.py`) and the ML result objects change together. In one
   repository a schema change, its UI change and its tests land atomically and are reviewed
   together. With separate repositories, versions drift and the frozen contract breaks silently.
2. **Evaluated code is served code.** `ml/` is imported directly by the training scripts,
   `ml.evaluation.evaluate` and the API. There is no packaging or publishing step that could let
   the API run a different preprocessing or decision version than the one evaluated (see ADR-014).
3. **Reproducibility of a scientific result.** Dataset config, calibration objectives, generated
   reports, model code and UI live under one commit hash, so a reported result can be traced back
   to exactly the code and configuration that produced it.
4. **Hackathon and team ergonomics.** Five people work in one clone with one README, one issue
   tracker and one set of commands. Each person still owns a clearly bounded directory (ML,
   API, web), so parallel work rarely conflicts.
5. **No microservice sprawl** (a spec principle). It is still one Python service and one web
   app, each with its own toolchain (`.venv` / `pnpm`) and its own Docker image, so either can be
   deployed independently later.

**Rejected.** *Next.js at the root with Python beside it*: this mixes two toolchains' config files
at the root and puts ESLint/TypeScript in the way of the Python tree. *Polyrepo*: it adds
cross-repository versioning for a contract that is still evolving. *A pnpm workspace at the root*:
with only one JS package it adds indirection, and the web Docker build is simpler with
`apps/web` as a self-contained context.

## ADR-002: Dataset: Mikrobat primary, DIMPSAR as far-OOD

**Decision.** Mikrobat (github.com/alamraya/Mikrobat, pinned commit `063b0e5`) is the only training
and reference dataset. A seeded subset of DIMPSAR (Hugging Face, pinned revision, SHA-256
verified) is used only as far-out-of-distribution negatives.

**Why.** Microscopic simplicia fragments match the pharmacognostic screening problem
(`docs/Data_Set.md`). The alternatives were verified during the dataset-lock step:
- SimpliScopeX: the public repository contains a trained model and a Streamlit app, not images.
- MPalyn: the training repository has code only, and the database site returned HTTP 404.
- DIMPSAR: CC BY 4.0, 5,945 images and 40 classes, verified on download. These are field-leaf
  photographs, a different modality.

**Constraints recorded.** Mikrobat has only two species (`sirih`, `sirih_merah`). The repository
does not include a license file. DIMPSAR's `Betel` class is *Piper betle*, the same species as
Mikrobat's `sirih`. DIMPSAR is therefore a **modality** far-OOD set (field photograph vs. micrograph),
not a set of different species, and it is never added as a training class.

## ADR-003: Duplicate and leakage control

**Decision.**
1. Exact duplicates: SHA-256. Within a class, one copy is kept. A byte-identical image filed under
   more than one class (53 found) is a label conflict and goes to the ambiguity probe (ADR-004).
2. Near-duplicates: 64-bit DCT pHash. Two images are linked when their Hamming distance is
   **below the 0.01th percentile of cross-class pair distances**, and links are merged transitively
   into groups. Groups, not images, are the split unit. A group spanning both classes is a label
   conflict and is excluded.

**Why this bound.** Images of different species cannot come from the same physical specimen, so
cross-class distances form an empirical null distribution for "distinct images". This turns an
otherwise arbitrary Hamming cut-off into an explicit false-link rate. It resolved to `< 16` on
88,920 cross-class pairs, which linked 24 pairs and excluded 4 cross-class near-duplicate groups.
One of those was visually confirmed to be the same micrograph filed under both species.

**Rejected after measurement.**
- *Fixed pHash cut-off (8 or 10)*: visual checks found true duplicates at 10–16 and a distinct
  pair at 10.
- *Pixel NCC*: a confirmed duplicate scored 0.75, while non-duplicate candidate pairs scored up
  to 0.91.
- *DINOv2 cosine*: semantic rather than copy-specific; confirmed duplicates scored 0.88–0.90,
  while a visually different pair scored 0.97.
- *ORB + RANSAC inliers*: degenerate transforms gave unrelated cross-species pairs over 250
  inliers.

**Residual risk.** Overlapping crops of one source micrograph can have large pHash distances and
may still cross splits. This is stated in the dataset report as possibly making test metrics
optimistic.

## ADR-004: Held-out known material and the ambiguity probe

**Decision.** For each class, the smallest fragment type with ≥ 30 eligible images is withheld
from training and from the reference library: `idioblas berupa sel minyak` (sirih) and
`xilem dengan noktah` (sirih_merah). These are **held-out known-material samples**, not "unknown
species". The 53 label-conflict images form a separate `ambiguity_probe` set.

**Why.** With two species, held-out-*species* calibration is impossible. Held-out fragment types
measure whether genuine material of a known species that the library does not represent is
wrongly rejected or wrongly accepted. The rule is deterministic rather than hand-picked. The
remaining groups are split 70/15/15, stratified by class × fragment type.

**Consequence (reported, not tuned away).** Held-out classification accuracy is 49.0% against
88.9% on test. Species and fragment type are confounded in Mikrobat: after curation the two
classes share no fragment-type label. See the evaluation findings.

## ADR-005: Preprocessing

**Decision.** Convert to grayscale, pad to square with the mean intensity, bicubic resize to 224,
then ImageNet normalisation, versioned as `preprocess-v1`. EXIF orientation is applied on decode.

**Why.**
- *Grayscale*: all 716 Mikrobat files store identical R/G/B channels (measured). Colour carries no
  reference information, and without this step colour would become a shortcut for rejecting
  colour photographs.
- *No crop*: the spec's microscopy cropping policy, because diagnostic structures can sit at the
  frame edge.
- *Mean padding*: keeps the aspect ratio without inventing structure (reflection padding would
  mirror cell walls).
- *Versioned*: the version is part of the embedding fingerprint (ADR-006), so an incompatible
  change is detected at load time.

## ADR-006: Frozen DINOv2 encoder

**Decision.** DINOv2 ViT-S/14 (`facebook/dinov2-small`, pinned revision) via `transformers`,
frozen and in inference mode. The embedding is the layer-normalised CLS token, L2-normalised
(384-d). Weights are saved to `models/pretrained/dinov2_vits14/` with a `SOURCE.json` receipt and
always loaded with `local_files_only=True`.

**Why.** DINOv2 features work directly with linear classifiers and nearest-neighbour retrieval.
ViT-S keeps CPU inference practical (about 12 images/s batched on the development laptop). Local,
pinned weights make the demo work offline. Because the weights directory describes itself, the
API does not need the training config.

**Embedding fingerprint.** `encoder@revision|preprocessing-version|dim` is written into every
artifact. `ScreeningPipeline` refuses to run when the classifier, index and calibration
fingerprints differ.

## ADR-007: Logistic-regression classifier

**Decision.** scikit-learn `LogisticRegression` with balanced class weights. C is chosen from
{0.01 … 10⁴} by **validation log-loss**, and the final model is fitted on the training split only.

**Why.** Log-loss is a proper scoring rule, so the chosen model has honest probabilities, which
matters because confidence feeds a calibrated threshold. The optimum (C = 100) is interior to the
grid; the grid was widened once when the optimum sat on its edge. Fitting on train only keeps the
validation split unseen for unknown and decision calibration.

## ADR-008: Reference library and FAISS

**Decision.** For each class, quality-acceptable training images are clustered with k-means
(k = 20, the spec's 5–20 range), and the real image nearest each centroid is kept: 40 references.
Index: FAISS `IndexFlatIP` over L2-normalised vectors (cosine similarity). `build_index` refuses
to index any image whose id or SHA-256 appears in another split.

**Why.** A small curated atlas is what a reviewer can inspect on the result page. Medoids are
real, representative and diverse images. Exact search is deterministic and sub-millisecond at
this size, so an approximate index would only add recall loss.

## ADR-009: Reproducible artifacts and what git tracks

**Decision.** All external inputs are pinned (Mikrobat commit, DIMPSAR revision + SHA-256, DINOv2
revision) and every choice lives in `ml/configs/pipeline-v1.json`. `python -m scripts.run_pipeline`
rebuilds everything. Git tracks source, configs, docs and generated **reports** (`docs/reports/`).
`data/*` and `models/*` are git-ignored except their READMEs.

**Why.** The generated data and models are large (weights and embeddings), reproducible, and
include Mikrobat images that must not be redistributed. The human-readable reports give the
audit trail in git. Text line endings are normalised to LF via `.gitattributes` for Docker and
cross-platform builds.

## ADR-010: Unknown detection and calibration

**Decision.**
- *Signal*: `distance = 1 − mean cosine similarity to the k = 5 nearest references`.
- *Threshold*: minimise DIMPSAR false acceptance subject to known acceptance ≥ 95% on Mikrobat
  validation; within the optimal interval, take the midpoint up to the next OOD distance.
- *Known boundary*: the distance meeting the 95% target. Between the boundary and the threshold
  the status is `UNCERTAIN`.
- *Risk*: logistic P(OOD | distance), fitted with balanced classes on a standardised feature.
- *Data*: calibrated on the validation split and DIMPSAR calibration classes; evaluated on the
  test set, held-out known material and **class-disjoint** DIMPSAR evaluation classes.

**Why.** The mean over k neighbours measures density in reference space, while the single best
match stays in the retrieval evidence. The objective encodes the spec's priority (no false
acceptance) as an explicit constraint rather than a guessed number. Calibrated values: threshold
0.560, boundary 0.431. Known validation distances top out at 0.535; DIMPSAR starts at 0.689.

**Limitation (verbatim in every result).** Because Mikrobat contains only two species, true
held-out-species OOD calibration was not possible. See `models/classifiers/calibration.json`.

## ADR-011: Image quality

**Decision.** Six checks: resolution, focus (Laplacian variance), under- and over-exposure,
clipped pixels, and specimen visibility (share of 8×8 tiles with detail). They are measured on a
canvas whose short side is 224 px. Bounds are percentile 1 / 99 of the Mikrobat training images;
`min_side = 224` is technical (the encoder input size). Any failed check makes quality `DEGRADED`.

**Why.** "Calibrated" means "atypical relative to the material the model was built from", not an
invented optical standard. Quality can only turn PASS into REVIEW and never changes identity.

## ADR-012: Evidence engine

**Decision.**
- *Agreement* is structural: HIGH when all top-k references share the predicted class, MEDIUM
  when the retrieval vote agrees but is mixed, LOW when retrieval favours another class.
- *Evidence strength* is the geometric mean of classifier confidence, reference support,
  (1 − unknown risk) and the quality pass fraction.

**Why.** There are no tuned agreement numbers. The geometric mean is low when any component is
low. Strength is shown as a descriptive index and deliberately **not** used by the decision
policy, which avoids a second, uncalibrated decision path. The wording always says "complementary
analyses of the same visual representation".

## ADR-013: Deterministic decision policy

**Decision.** Rules in order:
1. Unknown status UNKNOWN → `UNKNOWN`.
2. All five PASS criteria met → `PRELIMINARY_PASS`. The criteria are confidence, top reference
   similarity, unknown risk, HIGH agreement and ACCEPTABLE quality.
3. Otherwise → `REVIEW_REQUIRED`, and the reason lists every failed criterion.

Calibrated on validation (`models/configs/decision-v1.json`):
- `min_classifier_confidence`: smallest threshold with ≥ 95% selective accuracy.
- `min_reference_similarity`: keeps 95% of correctly classified samples.
- `max_unknown_risk`: the risk at the known boundary.

**Why.** Rules are reproducible, auditable and unit-tested, and each result shows every criterion
with observed and required values. On the test split all 24 PASS decisions were correct and all 8
misclassifications were stopped.

## ADR-014: One screening pipeline for API, evaluation and smoke test

**Decision.** `ml/inference/screening_pipeline.py::ScreeningPipeline` owns image → quality →
embedding → classifier → retrieval → unknown → evidence → decision → explanation. Its constructor
takes components (dependency injection), and `ScreeningPipeline.load()` reads artifacts once.

**Why.** The metrics in the evaluation report describe exactly the code the API serves. Injection
lets tests run the real logic with a lightweight encoder double.

## ADR-015: FastAPI service

**Decision.**
- *One service*, started once with artifacts loaded in the lifespan hook. Missing or mismatched
  artifacts start the API in **degraded mode**: `/health` explains the problem, `/analyze` returns
  503, and history stays browsable.
- *Endpoints under `/api/v1`*: `POST /analyze`, `GET /reference/{id}`, `GET /health` (spec), plus
  `GET /analyses`, `GET /analyses/{id}`, `GET /analyses/{id}/image` and
  `GET /reference/{id}/image`. The results page and history need these.
- *Upload safety*: allowed MIME types (415), size limit read with a one-byte overflow check (413),
  full decode validation (422), a 50 MP memory guard, server-generated UUIDs with id patterns, and
  path containment for reference images.
- *Storage*: stored sample images are re-encoded PNGs, never the raw upload bytes.
- *Persistence*: SQLite (`APP_STATE_DIR`), which needs no server, credentials or migrations.
- *Concurrency*: inference runs in a thread with a lock, because the encoder already uses every
  CPU core.
- *Schema additions*: `decision.thresholds`, `decision.checks`, `limitations`, `disclaimer` and
  `health.max_upload_mb` add transparency without changing the frozen blocks.

## ADR-016: Next.js frontend

**Decision.** Next.js 16 App Router, TypeScript, Tailwind v4 tokens (`app/globals.css`), and
Phosphor icons via the `@phosphor-icons/react/ssr` entry, which works in both server and client
components.
- *Pages*: `/`, `/analyze`, `/results/[id]` (async `params`, as required in v16) and `/history`.
- *Data*: fetched client-side through `lib/api.ts` and `hooks/useApiResource.ts`, with explicit
  loading, error and retry states.
- *Fonts*: Geist bundled from the `geist` npm package, not `next/font/google`.
- *Images*: API images use `next/image` with `unoptimized`.

**Why.** The browser is the only API consumer, so there is one `NEXT_PUBLIC_API_BASE_URL` and no
server-to-server URL to keep in sync. Bundled fonts keep dev and build working offline, a spec
requirement. Next 16's image optimiser blocks local IP addresses (an SSRF guard), and the
micrographs are already small. The upload limit is read from `/health`, so it is not duplicated in
the client. Every result section shows the underlying numbers, thresholds and reference images.

## ADR-017: No LLM

**Decision.** Explanations are deterministic templates built only from structured results.

**Why.** The spec makes an LLM optional and forbids it from influencing any decision. Templates
cannot invent biological facts, keep the system fully offline, and avoid an unused integration
path.

## ADR-018: Testing strategy

**Decision.**
- *Python unit tests* for preprocessing, quality, classifier, retrieval, unknown detector,
  evidence and decision engines, and dataset helpers.
- *A synthetic end-to-end pipeline* (`tests/helpers.py`) built with the real training and
  calibration functions and a deterministic encoder double. It drives pipeline and API tests
  (upload → inference → persistence, invalid files, unknown handling, degraded mode, CORS).
- *Integration tests on the real artifacts*, marked `artifacts`, run when the pipeline has been
  built and are skipped otherwise.
- *Frontend Vitest + Testing Library tests* for upload, loading, success, error and result
  rendering, using fixtures captured from the real API.

## ADR-019: Docker

**Decision.** `docker/api.Dockerfile` installs CPU-only torch first and mounts `./models`
(read-only) and `./data`. `docker/web.Dockerfile` builds a multi-stage Next.js `standalone`
server and passes the browser-facing API URL as a build argument. `docker-compose.yml` starts the
web service only after the API health check passes.

**Why.** Artifacts are reproducible and partly non-redistributable (ADR-009). CPU wheels avoid
multi-GB CUDA layers. `NEXT_PUBLIC_*` values are inlined at build time.
