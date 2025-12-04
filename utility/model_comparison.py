"""
Utilities to compare the trained RL threshold-tuning policy against the
frozen YOLO baseline on the same dataset.

Typical usage (after configuring .env):
    python -m utility.model_comparison --model ppo_model.zip
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO
from ultralytics import YOLO

from utility.dataset import load_custom_dataset, load_pascal_voc2007
from utility.evaluation import EpisodeStats, evaluate_policy, summarize_stats
from utility.metrics import compute_iou


def _prepare_dataset(limit_override: int | None = None) -> Tuple[Iterable, str]:
    """
    Load dataset based on .env configuration.

    - If IMAGE_DIR and LABEL_DIR are set and exist, use the custom YOLO-format data.
    - Otherwise, fall back to Pascal VOC 2007 (auto-download if missing).
    """

    load_dotenv()
    data_limit = limit_override if limit_override is not None else int(os.getenv("RL_DATA_LIMIT", "500"))
    image_dir = os.getenv("IMAGE_DIR")
    label_dir = os.getenv("LABEL_DIR")
    voc_root = os.getenv("VOC_ROOT", "./data/voc")

    if image_dir and Path(image_dir).exists():
        dataset = load_custom_dataset(
            image_dir=image_dir,
            label_dir=label_dir or image_dir,
            limit=data_limit,
        )
        return dataset, f"custom({image_dir})"

    dataset = load_pascal_voc2007(root=voc_root, limit=data_limit, download=True)
    return dataset, f"voc2007({voc_root})"


def _evaluate_baseline(dataset: Sequence, thresholds: Sequence[float]) -> dict:
    """
    Evaluate the frozen YOLO model over a set of fixed confidence thresholds.
    """

    model = YOLO("yolov8n.pt")
    per_threshold: List[dict] = []

    for t in thresholds:
        ious: List[float] = []
        for img, gt_boxes in dataset:
            img_uint8 = (img * 255).clip(0, 255).astype("uint8")
            preds = model.predict(img_uint8, conf=float(t), verbose=False)[0]
            pred_boxes = preds.boxes.xyxy.cpu().numpy() if len(preds.boxes) else np.empty((0, 4))
            ious.append(float(compute_iou(pred_boxes, gt_boxes)))

        mean_iou = float(np.mean(ious)) if ious else 0.0
        std_iou = float(np.std(ious)) if ious else 0.0
        per_threshold.append({"threshold": float(t), "mean_iou": mean_iou, "std_iou": std_iou})

    best = max(per_threshold, key=lambda x: x["mean_iou"]) if per_threshold else {"threshold": None, "mean_iou": 0.0, "std_iou": 0.0}
    return {"per_threshold": per_threshold, "best": best}


def compare_models(
    model_path: str,
    thresholds: Sequence[float] | None = None,
    episodes: int | None = None,
    deterministic: bool = True,
    dataset_limit: int | None = None,
) -> dict:
    """
    Run RL policy rollouts and fixed-threshold baselines, return a summary dict.
    """

    if thresholds is None:
        thresholds = (0.3, 0.4, 0.5, 0.6, 0.7)

    dataset, dataset_name = _prepare_dataset(limit_override=dataset_limit)
    episodes = episodes or int(os.getenv("RL_EVAL_EPISODES", "5"))

    rl_model = PPO.load(model_path)
    rl_stats: List[EpisodeStats] = evaluate_policy(
        model=rl_model,
        dataset=dataset,
        episodes=episodes,
        deterministic=deterministic,
    )
    rl_summary = summarize_stats(rl_stats)

    baseline = _evaluate_baseline(dataset, thresholds)
    best_baseline = baseline["best"]

    improvement = rl_summary["mean_final_iou"] - best_baseline["mean_iou"]

    return {
        "dataset": dataset_name,
        "episodes": episodes,
        "rl": rl_summary,
        "baseline": baseline,
        "improvement_over_best_baseline": float(improvement),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RL policy vs frozen YOLO baseline.")
    parser.add_argument("--model", required=True, help="Path to the trained PPO policy (.zip).")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="Confidence thresholds to evaluate the baseline YOLO model.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Number of RL evaluation episodes.")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions during evaluation.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = compare_models(
        model_path=args.model,
        thresholds=args.thresholds,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        dataset_limit=None,
    )
    # Lightweight text output; relying on CLI styling for readability.
    print("Dataset:", results["dataset"])
    print("Episodes:", results["episodes"])
    print("RL mean_final_iou:", results["rl"]["mean_final_iou"])
    print("RL std_final_iou:", results["rl"]["std_final_iou"])
    print("Best baseline threshold:", results["baseline"]["best"]["threshold"])
    print("Best baseline mean_iou:", results["baseline"]["best"]["mean_iou"])
    print("Improvement over best baseline:", results["improvement_over_best_baseline"])
    print("Per-threshold baseline results:")
    for row in results["baseline"]["per_threshold"]:
        print(f"  t={row['threshold']:.2f} mean_iou={row['mean_iou']:.4f} std={row['std_iou']:.4f}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
