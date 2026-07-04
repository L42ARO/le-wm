import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def get_results_path(cfg: DictConfig) -> Path:
    policy_name = Path(cfg.policy).stem if cfg.policy != "random" else "random"
    return Path(__file__).parent / "models" / cfg.model_name / "evals" / policy_name


def get_policy_path(cfg: DictConfig) -> Path | str:
    if cfg.policy == "random":
        return "random"
    return Path(__file__).parent / "models" / cfg.model_name / "weights" / cfg.policy


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episode_index_column(dataset):
    for col_name in ("episode_idx", "ep_idx"):
        try:
            dataset.get_col_data(col_name)
            return col_name
        except Exception:
            continue
    raise KeyError("Dataset is missing both 'episode_idx' and 'ep_idx' columns.")


def get_episodes_length(dataset, episodes):
    col_name = get_episode_index_column(dataset)
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset_root = dataset_path / "datasets"
    candidates = [dataset_name]
    if "." not in Path(dataset_name).name:
        candidates = [
            f"{dataset_name}.lance",
            f"{dataset_name}.h5",
            dataset_name,
        ]

    resolved_name = None
    for candidate in candidates:
        if Path(candidate).is_absolute() or (dataset_root / candidate).exists():
            resolved_name = candidate
            break

    dataset = swm.data.load_dataset(
        resolved_name or dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def build_world_and_process(cfg: DictConfig) -> tuple[Any, Any, dict]:
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    return world, dataset, {"process": process, "transform": transform}


def build_policy(cfg: DictConfig, process: dict, transform: dict):
    policy_path = get_policy_path(cfg)
    if policy_path == "random":
        return swm.policy.RandomPolicy()

    model = swm.wm.utils.load_pretrained(str(policy_path))
    model = model.to("cuda")
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    return swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )


def get_eval_starts(cfg: DictConfig, dataset) -> tuple[list[int], list[int]]:
    col_name = get_episode_index_column(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    episode_idx_all = dataset.get_col_data(col_name)
    step_idx_all = dataset.get_col_data("step_idx")
    eval_episodes = episode_idx_all[random_episode_indices]
    eval_start_idx = step_idx_all[random_episode_indices]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    return eval_episodes.tolist(), eval_start_idx.tolist()


def write_results(cfg: DictConfig, results_dir: Path, metrics: dict, elapsed: float):
    results_file = results_dir / cfg.output.filename
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with results_file.open("a") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {elapsed} seconds\n")


def evaluate_cfg(cfg: DictConfig) -> dict:
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    world, dataset, resources = build_world_and_process(cfg)
    policy = build_policy(cfg, resources["process"], resources["transform"])
    results_path = get_results_path(cfg)
    eval_episodes, eval_start_idx = get_eval_starts(cfg, dataset)

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx,
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes,
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()

    print(metrics)
    write_results(cfg, results_path, metrics, end_time - start_time)
    return metrics


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    evaluate_cfg(cfg)


if __name__ == "__main__":
    run()
