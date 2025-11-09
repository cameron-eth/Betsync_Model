#!/usr/bin/env python3
"""
NBA Predictions Runner
---------------------
Lightweight entry point that loads the trained NBA hybrid model and publishes
predictions without retraining or touching the NFL pipeline.
"""

import sys
import time
from pathlib import Path

# Ensure we can import from src/
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root / 'src'))

from src.nba_hybrid_pipeline import NBAHybridPipeline  # noqa: E402
from src.nba_utils import print_section_header  # noqa: E402


def main() -> int:
    print_section_header("NBA DAILY PREDICTIONS")
    start = time.time()

    try:
        pipeline = NBAHybridPipeline()
        pipeline.run_full_pipeline()
    except Exception as exc:  # pragma: no cover - surfaced in CI logs
        print(f"❌ NBA prediction run failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    elapsed = time.time() - start
    print(f"\n✅ NBA predictions completed in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

