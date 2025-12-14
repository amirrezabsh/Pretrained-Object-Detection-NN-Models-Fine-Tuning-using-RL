"""
Utilities to compare the trained RL threshold-tuning policy against the
frozen YOLO baseline on the same dataset.

Typical usage (after configuring .env):
    python -m utility.model_comparison --model ppo_model.zip
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO
from ultralytics import YOLO

from utility.dataset import load_custom_dataset, load_pascal_voc2007
from utility.evaluation import EpisodeStats, evaluate_policy, summarize_stats
from utility.logging_utils import setup_logging
from utility.metrics import compute_iou
from utility.torch_utils import get_default_device

logger = logging.getLogger(__name__)

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
        logger.info("Using custom dataset image_dir=%s limit=%s", image_dir, data_limit)
        return dataset, f"custom({image_dir})"

    dataset = load_pascal_voc2007(root=voc_root, limit=data_limit, download=True)
    logger.info("Using VOC2007 root=%s limit=%s", voc_root, data_limit)
    return dataset, f"voc2007({voc_root})"


def _evaluate_baseline(
    dataset: Sequence,
    thresholds: Sequence[float],
    device,
    max_det: int = 200,
    nms_iou: float = 0.7,
) -> dict:
    """
    Evaluate the frozen YOLO model over a set of fixed confidence thresholds.
    """

    model = YOLO("yolov8n.pt").to(device)
    per_threshold: List[dict] = []

    for t in thresholds:
        ious: List[float] = []
        logger.info("Evaluating baseline at threshold=%.2f", t)
        for img, gt_boxes in dataset:
            img_uint8 = (img * 255).clip(0, 255).astype("uint8")
            preds = model.predict(
                img_uint8,
                conf=float(t),
                device=device,
                max_det=max_det,
                iou=nms_iou,
                verbose=True,
            )[0]
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

    policy_device = get_default_device(env_var="RL_DEVICE") if os.getenv("RL_DEVICE") else get_default_device()
    detector_device = get_default_device(env_var="DETECTOR_DEVICE")
    max_det = int(os.getenv("COMPARE_MAX_DET", "200"))
    nms_iou = float(os.getenv("COMPARE_NMS_IOU", "0.7"))
    if thresholds is None:
        thresholds = (0.3, 0.4, 0.5, 0.6, 0.7)

    dataset, dataset_name = _prepare_dataset(limit_override=dataset_limit)
    episodes = episodes or int(os.getenv("RL_EVAL_EPISODES", "5"))

    logger.info("Running comparison on dataset=%s episodes=%d thresholds=%s", dataset_name, episodes, thresholds)
    rl_model = PPO.load(model_path, device=policy_device)
    env_kwargs = {
        "max_delta": float(os.getenv("RL_MAX_DELTA", "0.2")),
        "max_steps": int(os.getenv("RL_MAX_STEPS", "15")),
    }
    rl_stats: List[EpisodeStats] = evaluate_policy(
        model=rl_model,
        dataset=dataset,
        episodes=episodes,
        deterministic=deterministic,
        device=policy_device,
        detector_device=detector_device,
        env_kwargs=env_kwargs,
    )
    rl_summary = summarize_stats(rl_stats)

    baseline = _evaluate_baseline(
        dataset,
        thresholds,
        device=detector_device,
        max_det=max_det,
        nms_iou=nms_iou,
    )
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
    setup_logging()
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
