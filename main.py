"""Run the full pipeline: download -> preprocess -> train -> evaluate.

Examples
    python main.py                                   # stages from config/config.yaml
    python main.py --stages preprocess train_lane    # a subset, in this order
    python main.py --skip download                   # everything except download
    python main.py --set lane.model.name=unet --set detection.train.epochs=20
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from waymo_training.cli import build_parser, load_from_args  # noqa: E402
from waymo_training.pipeline import STAGES, run_stages  # noqa: E402


def main() -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), help="stages to run")
    parser.add_argument("--skip", nargs="+", choices=list(STAGES), default=[], help="stages to skip")
    args = parser.parse_args()
    cfg = load_from_args(args)
    stages = args.stages or [str(s) for s in cfg.pipeline.stages]
    run_stages(cfg, [s for s in stages if s not in args.skip])
    return 0


if __name__ == "__main__":
    sys.exit(main())
