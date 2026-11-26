"""Canonical BAGI entry point: smoke, native training, and frozen evaluation."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m bagi")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("smoke", help="Synthetic architecture contract (default)")
    native = commands.add_parser("workshop", help="Native diagnostic using the shared runtime")
    native.add_argument("mode", choices=["train", "evaluate"])
    native.add_argument("--output", required=True, type=Path)
    native.add_argument("--episodes", required=True, type=int)
    native.add_argument("--first-seed", required=True, type=int)
    native.add_argument("--model-seed", default=0, type=int)
    native.add_argument("--max-steps", default=100, type=int)
    native.add_argument("--batch-size", default=64, type=int)
    native.add_argument("--epsilon", default=0.2, type=float)
    native.add_argument("--layout", choices=["default", "mirror_x"], default="default")
    native.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "workshop":
        from bagi.workshop_runtime import run_workshop

        result = run_workshop(
            args.output, mode=args.mode, episodes=args.episodes, first_seed=args.first_seed,
            model_seed=args.model_seed, max_steps=args.max_steps, batch_size=args.batch_size,
            epsilon=args.epsilon, layout=args.layout, checkpoint=args.checkpoint,
        )
    else:
        from bagi.experiments.general_brain_smoke import run

        result = run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
