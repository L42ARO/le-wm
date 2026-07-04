import argparse
import ast
import contextlib
import csv
import io
import re
from pathlib import Path
from typing import Iterable

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from tqdm import tqdm

from eval import evaluate_cfg, get_results_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pusht")
    parser.add_argument("--clean-videos", action="store_true")
    parser.add_argument("--clean-logs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_known_args()


def build_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    config_dir = str((Path(__file__).parent / "config" / "eval").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name, overrides=overrides)


def list_policies(weights_dir: Path) -> list[str]:
    policies = sorted(path.name for path in weights_dir.glob("*.pt"))
    if not policies:
        raise FileNotFoundError(f"No .pt policies found under {weights_dir}")
    return policies


def clean_videos(results_dir: Path):
    for video_path in results_dir.glob("env_*.mp4"):
        video_path.unlink()


def get_results_file(cfg: DictConfig) -> Path:
    return get_results_path(cfg) / cfg.output.filename


def parse_saved_metrics(results_file: Path) -> dict:
    text = results_file.read_text()
    matches = re.findall(r"metrics:\s*(\{.*?\})\s*\nevaluation_time:", text, flags=re.DOTALL)
    if not matches:
        raise ValueError(f"Could not parse metrics from {results_file}")

    metrics_text = matches[-1]
    success_match = re.search(r"'success_rate':\s*([0-9eE.+-]+)", metrics_text)
    episodes_match = re.search(r"'episode_successes':\s*array\((\[.*?\])\)", metrics_text, flags=re.DOTALL)
    if success_match is None or episodes_match is None:
        raise ValueError(f"Could not parse success data from {results_file}")

    return {
        "success_rate": float(success_match.group(1)),
        "episode_successes": ast.literal_eval(episodes_match.group(1)),
    }


def get_csv_row(policy: str, metrics: dict) -> list[object]:
    episode_successes = metrics["episode_successes"]
    if hasattr(episode_successes, "tolist"):
        episode_successes = episode_successes.tolist()
    return [Path(policy).stem, metrics["success_rate"], *episode_successes]


def get_csv_header(num_eval: int) -> list[str]:
    return ["policy", "success_rate", *[f"env_{i + 1}" for i in range(num_eval)]]


def run_policy_eval(cfg: DictConfig, clean_logs: bool) -> dict:
    if not clean_logs:
        return evaluate_cfg(cfg)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return evaluate_cfg(cfg)


def write_csv(csv_path: Path, rows: Iterable[list[object]], num_eval: int):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(get_csv_header(num_eval))
        writer.writerows(rows)


def main():
    args, overrides = parse_args()
    base_cfg = build_cfg(args.config_name, overrides)

    model_name = base_cfg.get("model_name")
    if model_name in (None, "???"):
        raise ValueError("model_name is required. Pass it as a Hydra override, e.g. model_name=cube")

    weights_dir = Path(__file__).parent / "models" / model_name / "weights"
    policies = list_policies(weights_dir)

    rows = []
    iterator = tqdm(policies, desc=f"Evaluating {model_name}") if args.clean_logs else policies
    for policy in iterator:
        cfg = build_cfg(args.config_name, [*overrides, f"policy={policy}"])
        results_file = get_results_file(cfg)
        ran_eval = False
        if results_file.exists() and not args.overwrite:
            metrics = parse_saved_metrics(results_file)
        else:
            metrics = run_policy_eval(cfg, clean_logs=args.clean_logs)
            ran_eval = True
        rows.append(get_csv_row(policy, metrics))
        if args.clean_videos and ran_eval:
            clean_videos(get_results_path(cfg))

    csv_path = Path(__file__).parent / "models" / model_name / "evals" / "summary.csv"
    write_csv(csv_path, rows, base_cfg.eval.num_eval)


if __name__ == "__main__":
    main()
