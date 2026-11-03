from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadow_occ.cli import run_make_labels


def main() -> None:
    parser = argparse.ArgumentParser("Generate shadow pseudo labels.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataroot")
    parser.add_argument("--occ-root")
    parser.add_argument("--geometry-source", choices=["occ3d", "nuscenes_lidar"])
    parser.add_argument("--output-root")
    parser.add_argument("--scene", action="append", default=None)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=1)
    run_make_labels(parser.parse_args())


if __name__ == "__main__":
    main()
