from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadow_occ.cli import run_corr


def main() -> None:
    parser = argparse.ArgumentParser("Evaluate shadow length / height correlation.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--label-root")
    parser.add_argument("--output-root")
    run_corr(parser.parse_args())


if __name__ == "__main__":
    main()
