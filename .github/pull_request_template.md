## What changes

<!-- One paragraph: what this does and why. -->

## Evidence

<!-- If a reported number changes, name the regenerated report in docs/reports/. -->

- [ ] `pytest`, `ruff check`, `ruff format --check`
- [ ] `pnpm lint`, `pnpm typecheck`, `pnpm test`, `pnpm build`
- [ ] `README.md` and `docs/` updated

## Screening rules

- [ ] No hard-coded predictions, confidences, counts or thresholds
- [ ] Any model-recipe change went through a pre-registered ladder (ADR-022)
- [ ] Wording keeps classifier and retrieval as complementary analyses of one representation
