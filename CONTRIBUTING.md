# Contributing to HerbaScope

HerbaScope makes claims about real material, so the rules below are not style preferences: they
are what keeps a screening result trustworthy.

## Non-negotiables

1. **No fabricated numbers.** Every value a user sees comes from an artifact the pipeline produced.
   Never hard-code a prediction, a confidence, a dataset count or a threshold.
2. **No hand-chosen thresholds.** Thresholds are calibrated by `ml/training/calibrate_*.py` against
   the objectives in `ml/configs/pipeline.json` and written to `models/`.
3. **Model changes go through a pre-registered ladder.** Commit the experiment
   (`ml/configs/experiments-v*.json`) *before* running it, and adopt a change only by its
   cross-validation rule, never by test or held-out accuracy (ADR-022).
4. **Classifier and retrieval are "complementary analyses of the same visual representation"**,
   never "independent evidence".
5. **Held-out Mikrobat fragment groups are held-out known material**, never "unknown species".
   DIMPSAR is a far-out-of-distribution negative set, never a training class.
6. **No LLM in any decision path.** Explanations are deterministic templates.
7. **Keep `README.md` and `docs/` current with every change**, and leave no dead code.

## Getting set up

See the quick start in [`README.md`](README.md). In short: Python 3.12 virtual environment at the
repository root, `pnpm install` in `apps/web`, then `python -m scripts.run_pipeline` once to build
a model release.

## Before you open a pull request

```bash
.venv/Scripts/python -m pytest          # ML + API tests (use .venv/bin on macOS/Linux)
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m ruff format --check .
pnpm --dir apps/web lint
pnpm --dir apps/web typecheck
pnpm --dir apps/web test
pnpm --dir apps/web build
```

CI runs exactly these. Tests marked `artifacts` skip themselves when no trained release is present.

## Commits

Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`), imperative mood,
with a body that explains *why*. If a change alters a reported number, regenerate the affected
report in `docs/reports/` in the same commit.

## Architecture decisions

Anything structural gets an ADR in [`docs/Architecture_Decisions.md`](docs/Architecture_Decisions.md):
the decision, why, and what was rejected with the measurement that rejected it.
