"""Train and calibrate the ATOM-CBF local perception ensemble on ID data."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from atom_cbf import (
    ATOMPerceptionNetwork,
    CHECKPOINT_VERSION,
    calibrate_atom_margin,
    estimate_cone_lipschitz_constants,
    normalize_obstacle_targets,
    predict_ensemble_physical,
    ray_scan_target,
)
from testing_utils import generate_angles_rad


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the ATOM-CBF perception ensemble and calibrate its adaptive "
            "margin using only ID episodes."
        )
    )
    parser.add_argument(
        "transitions",
        nargs="+",
        type=Path,
        help="ID *_transitions.json files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ray-length", type=float, default=None)
    parser.add_argument("--rays-per-direction", type=int, default=None)
    parser.add_argument("--fov-degrees", type=float, default=None)
    parser.add_argument("--state-size", type=int, default=None)
    parser.add_argument("--stack-number", type=int, default=None)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--conv-channels", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--gamma-multiplier", type=float, default=1.0)
    parser.add_argument("--uncertainty-floor", type=float, default=1e-8)
    parser.add_argument("--d-safe", type=float, default=0.25)
    parser.add_argument("--d-safe-multiplier", type=float, default=3.0)
    parser.add_argument("--cbf-gain", type=float, default=1.5)
    parser.add_argument("--slack-penalty", type=float, default=100.0)
    parser.add_argument("--lipschitz-min-distance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cuda", type=int, default=-1)
    return parser.parse_args()


def companion_info_path(transitions_path: Path) -> Path:
    name = transitions_path.name
    if name.endswith("_transitions.json"):
        return transitions_path.with_name(name.replace("_transitions.json", "_info.json"))
    return transitions_path.with_suffix(".info.json")


def read_metadata(transitions_path: Path) -> dict[str, Any]:
    info_path = companion_info_path(transitions_path)
    if not info_path.exists():
        return {}
    with info_path.open("r", encoding="utf-8") as file:
        contents = json.load(file)
    return contents.get("metadata", {})


def resolve_sensor_config(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_metadata(args.transitions[0])
    other = metadata.get("other_config", {})
    test = metadata.get("test_config", {})

    rays_per_direction = args.rays_per_direction
    if rays_per_direction is None:
        rays_per_direction = other.get("rays_per_direction")
    ray_length = args.ray_length
    if ray_length is None:
        ray_length = other.get("raycast_length")
    state_size = args.state_size
    if state_size is None and "state_observation_size" in other:
        state_size = int(other["state_observation_size"]) - 1
    stack_number = args.stack_number
    if stack_number is None:
        train_path = test.get("train_config_path")
        if train_path and Path(train_path).exists():
            import yaml

            with Path(train_path).open("r", encoding="utf-8") as file:
                stack_number = (yaml.safe_load(file) or {}).get("input_stack")
    fov_degrees = args.fov_degrees
    if fov_degrees is None:
        fov_degrees = other.get("raycast_max_degrees", 90.0)

    missing = [
        name
        for name, value in (
            ("ray_length", ray_length),
            ("rays_per_direction", rays_per_direction),
            ("state_size", state_size),
            ("stack_number", stack_number),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Could not infer "
            + ", ".join(missing)
            + ". Pass the corresponding command-line options."
        )
    return {
        "ray_length": float(ray_length),
        "rays_per_direction": int(rays_per_direction),
        "ray_size": 2 * int(rays_per_direction) + 1,
        "fov_degrees": float(fov_degrees),
        "state_size": int(state_size),
        "stack_number": int(stack_number),
    }


def transition_observation(transition: Any) -> np.ndarray:
    if isinstance(transition, dict):
        if "obs" not in transition:
            raise ValueError("rich transition is missing the obs field")
        return np.asarray(transition["obs"], dtype=np.float32)
    values = np.asarray(transition, dtype=np.float32)
    if values.size < 3:
        raise ValueError("legacy transition is too short")
    return values[:-2]


def load_episode_scans(
    paths: Sequence[Path], sensor: dict[str, Any]
) -> list[np.ndarray]:
    expected_observation_size = (
        sensor["ray_size"] + sensor["state_size"]
    ) * sensor["stack_number"]
    ray_start = sensor["ray_size"] * (sensor["stack_number"] - 1)
    ray_end = ray_start + sensor["ray_size"]
    episodes: list[np.ndarray] = []

    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            contents = json.load(file)
        if not isinstance(contents, list):
            raise ValueError(f"{path} must contain a list of episodes")
        for episode in contents:
            scans: list[np.ndarray] = []
            for transition in episode:
                observation = transition_observation(transition)
                if observation.size != expected_observation_size:
                    raise ValueError(
                        f"{path} contains an observation of length {observation.size}; "
                        f"expected {expected_observation_size}"
                    )
                scans.append(np.clip(observation[ray_start:ray_end], 0.0, 1.0))
            if scans:
                episodes.append(np.stack(scans).astype(np.float32))
    if len(episodes) < 3:
        raise ValueError("at least three nonempty ID episodes are required")
    return episodes


def split_episodes(
    episodes: Sequence[np.ndarray],
    validation_fraction: float,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if validation_fraction <= 0 or calibration_fraction <= 0:
        raise ValueError("validation and calibration fractions must be positive")
    if validation_fraction + calibration_fraction >= 1:
        raise ValueError("validation and calibration fractions must sum to less than 1")

    indices = np.arange(len(episodes))
    np.random.default_rng(seed).shuffle(indices)
    validation_count = max(1, int(round(len(indices) * validation_fraction)))
    calibration_count = max(1, int(round(len(indices) * calibration_fraction)))
    while validation_count + calibration_count >= len(indices):
        if validation_count >= calibration_count and validation_count > 1:
            validation_count -= 1
        elif calibration_count > 1:
            calibration_count -= 1
        else:
            raise ValueError("not enough episodes for disjoint train/val/cal splits")

    validation_indices = indices[:validation_count]
    calibration_indices = indices[
        validation_count : validation_count + calibration_count
    ]
    train_indices = indices[validation_count + calibration_count :]
    return (
        [episodes[index] for index in train_indices],
        [episodes[index] for index in validation_indices],
        [episodes[index] for index in calibration_indices],
    )


def flatten_episodes(episodes: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(episodes, axis=0).astype(np.float32)


def make_targets(
    scans: np.ndarray,
    ray_angles: np.ndarray,
    ray_length: float,
    max_abs_bearing: float,
) -> tuple[np.ndarray, np.ndarray]:
    physical = np.stack(
        [ray_scan_target(scan, ray_angles, ray_length) for scan in scans]
    ).astype(np.float32)
    normalized = normalize_obstacle_targets(
        physical, ray_length, max_abs_bearing
    ).astype(np.float32)
    return normalized, physical


def evaluate_loss(
    model: ATOMPerceptionNetwork,
    scans: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, scans.shape[0], batch_size):
            prediction = model(scans[start : start + batch_size])
            squared_error = torch.square(
                prediction - targets[start : start + batch_size]
            ).sum(dim=1)
            total += float(squared_error.sum().item())
            count += int(squared_error.numel())
    return total / max(count, 1)


def train_model(
    model_index: int,
    train_episodes: Sequence[np.ndarray],
    validation_scans: np.ndarray,
    ray_angles: np.ndarray,
    sensor: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ATOMPerceptionNetwork, dict[str, Any]]:
    train_scans = flatten_episodes(train_episodes)
    max_abs_bearing = float(np.max(np.abs(ray_angles)))
    train_targets, _ = make_targets(
        train_scans, ray_angles, sensor["ray_length"], max_abs_bearing
    )
    validation_targets, _ = make_targets(
        validation_scans, ray_angles, sensor["ray_length"], max_abs_bearing
    )

    torch.manual_seed(args.seed + model_index)
    model = ATOMPerceptionNetwork(
        sensor["ray_size"], args.hidden_sizes, args.conv_channels
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    dataset = TensorDataset(
        torch.as_tensor(train_scans), torch.as_tensor(train_targets)
    )
    loader_generator = torch.Generator().manual_seed(args.seed + model_index)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    validation_scans_tensor = torch.as_tensor(validation_scans, device=device)
    validation_targets_tensor = torch.as_tensor(validation_targets, device=device)

    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    completed_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        for scan_batch, target_batch in loader:
            scan_batch = scan_batch.to(device)
            target_batch = target_batch.to(device)
            loss = torch.nn.functional.mse_loss(model(scan_batch), target_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        validation_loss = evaluate_loss(
            model,
            validation_scans_tensor,
            validation_targets_tensor,
            args.batch_size,
        )
        completed_epochs = epoch + 1
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no valid model state")
    model.load_state_dict(best_state)
    model.cpu().eval()
    return model, {
        "model_index": model_index,
        "epochs": completed_epochs,
        "best_validation_loss": float(best_loss),
        "training_transition_count": int(train_scans.shape[0]),
    }


def main() -> None:
    args = parse_cli()
    if args.ensemble_size < 2:
        raise ValueError("ensemble_size must be at least 2")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("training counts must be positive")
    if args.d_safe <= 0 or args.d_safe_multiplier <= 1 or args.cbf_gain <= 0:
        raise ValueError("invalid CBF geometry")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu"
    )
    sensor = resolve_sensor_config(args)
    ray_angles = np.asarray(
        generate_angles_rad(sensor["rays_per_direction"], sensor["fov_degrees"]),
        dtype=np.float64,
    )
    max_abs_bearing = float(np.max(np.abs(ray_angles)))
    episodes = load_episode_scans(args.transitions, sensor)
    train_episodes, validation_episodes, calibration_episodes = split_episodes(
        episodes,
        args.validation_fraction,
        args.calibration_fraction,
        args.seed,
    )
    validation_scans = flatten_episodes(validation_episodes)
    calibration_scans = flatten_episodes(calibration_episodes)
    _, calibration_targets = make_targets(
        calibration_scans,
        ray_angles,
        sensor["ray_length"],
        max_abs_bearing,
    )

    print(
        "ATOM-CBF ID split: "
        f"{len(train_episodes)} train episodes, "
        f"{len(validation_episodes)} validation episodes, "
        f"{len(calibration_episodes)} calibration episodes"
    )
    print(f"Training {args.ensemble_size} perception models on {device}...")
    models: list[ATOMPerceptionNetwork] = []
    model_stats: list[dict[str, Any]] = []
    for model_index in range(args.ensemble_size):
        model, stats = train_model(
            model_index,
            train_episodes,
            validation_scans,
            ray_angles,
            sensor,
            args,
            device,
        )
        models.append(model)
        model_stats.append(stats)
        print(
            f"  model {model_index + 1}/{args.ensemble_size}: "
            f"val={stats['best_validation_loss']:.6g}, "
            f"epochs={stats['epochs']}"
        )

    calibration_predictions = predict_ensemble_physical(
        models,
        calibration_scans,
        sensor["ray_length"],
        max_abs_bearing,
        device,
    )
    calibration = calibrate_atom_margin(
        calibration_predictions,
        calibration_targets,
        gamma_multiplier=args.gamma_multiplier,
        uncertainty_floor=args.uncertainty_floor,
    )
    activation_distance = args.d_safe * args.d_safe_multiplier
    lipschitz_min_distance = args.lipschitz_min_distance
    if lipschitz_min_distance is None:
        lipschitz_min_distance = args.d_safe + max(0.05, 0.1 * args.d_safe)
    if lipschitz_min_distance >= activation_distance:
        raise ValueError("lipschitz_min_distance must be below the activation distance")
    lipschitz = estimate_cone_lipschitz_constants(
        obstacle_radius=args.d_safe,
        min_distance=lipschitz_min_distance,
        max_distance=activation_distance,
        max_abs_bearing=max_abs_bearing,
        cbf_gain=args.cbf_gain,
    )

    checkpoint = {
        "format_version": CHECKPOINT_VERSION,
        "method": "ATOM-CBF",
        "config": {
            "input_dim": sensor["ray_size"],
            "hidden_sizes": list(args.hidden_sizes),
            "conv_channels": list(args.conv_channels),
            "ensemble_size": args.ensemble_size,
            "ray_length": sensor["ray_length"],
            "ray_angles": ray_angles.tolist(),
            "fov_degrees": sensor["fov_degrees"],
            "d_safe": args.d_safe,
            "d_safe_multiplier": args.d_safe_multiplier,
            "cbf_gain": args.cbf_gain,
            "slack_penalty": args.slack_penalty,
            "solver": "CLARABEL",
        },
        "model_state_dicts": [model.state_dict() for model in models],
        "calibration": calibration,
        "lipschitz": lipschitz,
        "training": {
            "seed": args.seed,
            "source_files": [str(path.resolve()) for path in args.transitions],
            "episode_counts": {
                "total": len(episodes),
                "train": len(train_episodes),
                "validation": len(validation_episodes),
                "calibration": len(calibration_episodes),
            },
            "transition_counts": {
                "train": int(sum(len(episode) for episode in train_episodes)),
                "validation": int(validation_scans.shape[0]),
                "calibration": int(calibration_scans.shape[0]),
            },
            "model_stats": model_stats,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Saved calibrated ATOM-CBF checkpoint to {args.output}")
    print(
        "Calibration: "
        f"phi={np.asarray(calibration['phi_cal'])}, "
        f"retained={calibration['retained_count']}/{calibration['sample_count']}, "
        f"L_Lgh={lipschitz['L_Lgh']:.6g}"
    )


if __name__ == "__main__":
    main()
