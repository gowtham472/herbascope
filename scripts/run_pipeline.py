"""Run the complete offline pipeline in dependency order.

  download_assets -> prepare_dataset -> extract_embeddings -> calibrate_quality ->
  train_classifier -> build_index -> calibrate_unknown -> calibrate_decision ->
  publish_release -> evaluate -> prepare_demo_cases

Usage:
  python -m scripts.run_pipeline                    # everything
  python -m scripts.run_pipeline --from build_index # resume from a step
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

STEPS = {
    "download_assets": "scripts.download_assets",
    "prepare_dataset": "ml.training.prepare_dataset",
    "extract_embeddings": "ml.training.extract_embeddings",
    "calibrate_quality": "ml.training.calibrate_quality",
    "train_classifier": "ml.training.train_classifier",
    "build_index": "ml.training.build_index",
    "calibrate_unknown": "ml.training.calibrate_unknown",
    "calibrate_decision": "ml.training.calibrate_decision",
    "publish_release": "ml.training.publish_release",
    "evaluate": "ml.evaluation.evaluate",
    "prepare_demo_cases": "scripts.prepare_demo_cases",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--from", dest="start", choices=list(STEPS), default="download_assets")
    names = list(STEPS)
    for name in names[names.index(parser.parse_args().start) :]:
        print(f"\n=== {name} ===", flush=True)
        started = time.perf_counter()
        status = importlib.import_module(STEPS[name]).main()
        if status:
            print(f"Step {name} failed with exit code {status}", file=sys.stderr)
            return status
        print(f"=== {name} done in {time.perf_counter() - started:.1f}s ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
