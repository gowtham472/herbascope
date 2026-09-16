# HerbaScope - agent guide

HerbaScope is a local-first, evidence-driven preliminary visual screening system for
microscopic medicinal-plant material. Read `README.md` first; design rationale lives in
`docs/Architecture_Decisions.md`.

## Layout (monorepo)

| Path | What | Toolchain |
|---|---|---|
| `apps/web/` | Next.js App Router frontend | pnpm, TypeScript, Tailwind, Phosphor Icons |
| `services/api/` | FastAPI service (`app` package) | Python 3.12 `.venv` at repo root |
| `ml/` | Shared ML library + training/evaluation modules | same `.venv` |
| `scripts/` | Orchestration entry points (download, pipeline, smoke test, demo cases) | same `.venv` |
| `tests/` | ML unit/integration tests | pytest |
| `docs/` | Specs, ADRs, generated reports (`docs/reports/`) | - |

Next.js-specific agent rules live in `apps/web/AGENTS.md` (managed by `next dev`). Read
the bundled docs in `apps/web/node_modules/next/dist/docs/` before writing frontend code.

## Non-negotiable rules

- Never fabricate predictions, confidences, dataset counts or thresholds. Every number
  shown to a user comes from a real artifact produced by the pipeline.
- Thresholds are calibrated by `ml/training/calibrate_*.py` against the objectives in
  `ml/configs/pipeline.json`; never hard-code them.
- Embedding-recipe changes go through a pre-registered ladder (`ml/configs/experiments-v*.json`,
  run with `python -m ml.experiments.run_experiments --config ...`) and are adopted only by its
  cross-validation rule, never by test or held-out accuracy.
- Classifier and reference retrieval are "complementary analyses of the same visual
  representation" - never describe them as independent evidence.
- Held-out Mikrobat fragment groups are "held-out known-material samples", never
  "unknown species". DIMPSAR is a far-OOD negative set, never a training class.
- No LLM participates in any decision. No Cloud Vision. No hardware dependencies.
- No dead code, no placeholder predictions. Keep `README.md` and `docs/` current with
  every change.

## Commands (run from repo root)

```bash
.venv/Scripts/python -m pytest          # Python tests (Windows path; use .venv/bin on macOS/Linux)
.venv/Scripts/python -m ruff check .    # Python lint
pnpm --dir apps/web test                # Frontend tests
pnpm --dir apps/web lint                # Frontend lint
```
