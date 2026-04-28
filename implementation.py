import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Neural Quantum Embedding experiments from a JSON config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Config file name in configs/ or a path to a JSON config file.",
    )
    return parser.parse_args()


def resolve_config_path(config_arg: str) -> Path:
    candidate = Path(config_arg).expanduser()
    search_paths = []

    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.append((Path.cwd() / candidate).resolve())
        search_paths.append((PROJECT_ROOT / candidate).resolve())
        search_paths.append((PROJECT_ROOT / "configs" / candidate.name).resolve())

    for path in search_paths:
        if path.is_file():
            return path

    searched = "\n".join(f"- {path}" for path in search_paths)
    raise FileNotFoundError(
        f"Could not find config '{config_arg}'. Searched:\n{searched}"
    )


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_run_dir(config: dict) -> Path:
    outdir = config.get("outdir", "results")
    run_dir = Path(outdir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    run_dir = build_run_dir(config)

    from runner import train_and_evaluate  # noqa: E402

    print(f"Using config: {config_path}")
    print(f"Writing outputs to: {run_dir}")
    train_and_evaluate(config, run_dir)


if __name__ == "__main__":
    main()
