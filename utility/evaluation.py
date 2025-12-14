from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from ultralytics import YOLO

from envionments.threshold_refinement import ThresholdRefinementEnv
from utility.metrics import compute_iou

logger = logging.getLogger(__name__)


@dataclass
class EpisodeStats:
    """
    Lightweight structure capturing one evaluation rollout.
    """

    episode: int
    steps: int
    final_iou: float
    reward_sum: float
    thresholds: List[float]
    ious: List[float]
    rewards: List[float]
    image_idx: int | None = None


def evaluate_policy(
    model: BaseAlgorithm,
    dataset: Sequence,
    episodes: int = 5,
    deterministic: bool = True,
    device=None,
    detector_device=None,
    env_kwargs: dict | None = None,
) -> List[EpisodeStats]:
    """
    Roll out a trained RL policy and collect summary statistics.
    """

    env_kwargs = env_kwargs or {}
    env = ThresholdRefinementEnv(dataset, device=device, detector_device=detector_device, **env_kwargs)
    stats: List[EpisodeStats] = []

    for ep in range(episodes):
        obs, _ = env.reset()
        thresholds = [float(obs[0])]
        done = False
        rewards = []
        ious = [float(env.prev_reward)]
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            thresholds.append(float(obs[0]))
            rewards.append(float(reward))
            ious.append(float(env.prev_reward))
            step += 1
            done = terminated or truncated

        stats.append(
            EpisodeStats(
                episode=ep,
                steps=step,
                final_iou=float(env.prev_reward),
                reward_sum=float(np.sum(rewards)),
                thresholds=thresholds,
                ious=ious,
                rewards=rewards,
                image_idx=int(getattr(env, "image_idx", -1)),
            )
        )

    return stats


def summarize_stats(stats: Iterable[EpisodeStats]) -> dict:
    """
    Aggregate statistics (mean/std) for quick notebook display.
    """

    final_ious = np.array([s.final_iou for s in stats], dtype=np.float32)
    reward_sums = np.array([s.reward_sum for s in stats], dtype=np.float32)
    steps = np.array([s.steps for s in stats], dtype=np.float32)

    summary = {
        "episodes": len(final_ious),
        "mean_final_iou": float(np.mean(final_ious)),
        "std_final_iou": float(np.std(final_ious)),
        "mean_return": float(np.mean(reward_sums)),
        "mean_steps": float(np.mean(steps)),
    }
    logger.info(
        "Evaluation summary: episodes=%d mean_final_iou=%.4f std_final_iou=%.4f mean_return=%.4f mean_steps=%.2f",
        summary["episodes"],
        summary["mean_final_iou"],
        summary["std_final_iou"],
        summary["mean_return"],
        summary["mean_steps"],
    )
    return summary


class EvalHistoryCallback(BaseCallback):
    """
    Lightweight callback to log evaluation mean IoU/return during training.
    """

    def __init__(
        self,
        eval_fn,
        eval_freq: int = 5000,
        name: str = "eval",
    ) -> None:
        super().__init__()
        self.eval_fn = eval_fn
        self.eval_freq = int(eval_freq)
        self.name = name
        self.history = []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            summary = self.eval_fn()
            summary["timesteps"] = self.num_timesteps
            self.history.append(summary)
            logger.info(
                "%s @ %d steps: mean_final_iou=%.4f mean_return=%.4f",
                self.name,
                self.num_timesteps,
                summary.get("mean_final_iou", 0.0),
                summary.get("mean_return", 0.0),
            )
        return True



def plot_threshold_trajectories(
    stats: Sequence[EpisodeStats],
    ax: plt.Axes | None = None,
    title: str = "Threshold trajectory per evaluation episode",
) -> plt.Axes:
    """
    Visualize how the policy adapts the threshold over timesteps.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    for s in stats:
        ax.plot(range(len(s.thresholds)), s.thresholds, marker="o", label=f"ep {s.episode}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Confidence threshold")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    return ax


def plot_iou_trajectories(
    stats: Sequence[EpisodeStats],
    ax: plt.Axes | None = None,
    title: str = "IoU trajectory per evaluation episode",
) -> plt.Axes:
    """
    Visualize IoU progression over steps (absolute IoU, not deltas).
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    for s in stats:
        ax.plot(range(len(s.ious)), s.ious, marker="o", label=f"ep {s.episode}")

    ax.set_xlabel("Step")
    ax.set_ylabel("IoU")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    return ax


def plot_reward_trajectories(
    stats: Sequence[EpisodeStats],
    ax: plt.Axes | None = None,
    title: str = "Reward (IoU delta) per step",
) -> plt.Axes:
    """
    Plot reward deltas per step to see when improvements happen.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    for s in stats:
        ax.plot(range(len(s.rewards)), s.rewards, marker="o", label=f"ep {s.episode}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (delta IoU)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    return ax


def _draw_boxes(
    ax: plt.Axes,
    boxes: np.ndarray,
    color: str,
    label: str,
    y_offset: float = 0.99,
) -> None:
    """
    Helper to overlay xyxy boxes on an axis.
    """

    import matplotlib.patches as patches

    for box in boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none", alpha=0.8
        )
        ax.add_patch(rect)
    ax.text(
        0.01,
        y_offset,
        label,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor=color),
    )


def visualize_policy_vs_baseline(
    model: BaseAlgorithm,
    dataset: Sequence,
    sample_indices: Sequence[int] | None = None,
    baseline_threshold: float = 0.5,
    baseline_nms_iou: float = 0.7,
    deterministic: bool = True,
    device=None,
    detector_device=None,
    env_kwargs: dict | None = None,
) -> plt.Figure:
    """
    Run the policy on a handful of images and compare detections vs a fixed-threshold baseline.

    Returns
    -------
    matplotlib.figure.Figure
        A figure with one row per sample: left=RL policy result, right=baseline YOLO.
    """

    if sample_indices is None:
        # Randomly choose a handful of unique samples each call
        k = min(4, len(dataset))
        sample_indices = np.random.choice(len(dataset), size=k, replace=False).tolist()

    env_kwargs = env_kwargs or {}
    env = ThresholdRefinementEnv(dataset, device=device, detector_device=detector_device, **env_kwargs)
    baseline_model = YOLO("yolov8n.pt").to(detector_device)

    n_rows = len(sample_indices)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for row, idx in enumerate(sample_indices):
        obs, _ = env.reset(options={"image_idx": idx})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        img, gt_boxes = dataset[idx]
        img_uint8 = (img * 255).clip(0, 255).astype("uint8")

        # RL policy prediction at final threshold
        rl_preds = env.model.predict(
            img_uint8,
            conf=float(env.current_threshold),
            iou=float(getattr(env, "current_nms_iou", baseline_nms_iou)),
            max_det=int(getattr(env, "current_max_det", 200)),
            device=detector_device,
            verbose=False,
        )[0]
        rl_boxes = rl_preds.boxes.xyxy.cpu().numpy() if len(rl_preds.boxes) else np.empty((0, 4))

        # Baseline fixed-threshold prediction
        base_preds = baseline_model.predict(
            img_uint8,
            conf=float(baseline_threshold),
            iou=float(baseline_nms_iou),
            device=detector_device,
            verbose=False,
        )[0]
        base_boxes = base_preds.boxes.xyxy.cpu().numpy() if len(base_preds.boxes) else np.empty((0, 4))

        # Plot RL result
        ax_rl = axes[row, 0]
        ax_rl.imshow(img_uint8)
        _draw_boxes(ax_rl, rl_boxes, color="tab:green", label=f"Policy t={env.current_threshold:.2f}", y_offset=0.99)
        _draw_boxes(ax_rl, gt_boxes, color="tab:orange", label="GT", y_offset=0.87)
        ax_rl.axis("off")
        ax_rl.set_title(
            f"Policy (idx {idx}) IoU={compute_iou(rl_boxes, gt_boxes):.3f} "
            f"dets={len(rl_boxes)} | GT={len(gt_boxes)}"
        )

        # Plot baseline result
        ax_base = axes[row, 1]
        ax_base.imshow(img_uint8)
        _draw_boxes(ax_base, base_boxes, color="tab:blue", label=f"Baseline t={baseline_threshold:.2f}", y_offset=0.99)
        _draw_boxes(ax_base, gt_boxes, color="tab:orange", label="GT", y_offset=0.87)
        ax_base.axis("off")
        ax_base.set_title(
            f"Baseline IoU={compute_iou(base_boxes, gt_boxes):.3f} "
            f"dets={len(base_boxes)} | GT={len(gt_boxes)}"
        )

    plt.tight_layout()
    return fig


def plot_final_iou_histogram(
    stats: Sequence[EpisodeStats],
    ax: plt.Axes | None = None,
    bins: int = 10,
    title: str = "Final IoU distribution",
) -> plt.Axes:
    """
    Histogram of per-episode final IoU to show spread/performance stability.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    final_ious = [s.final_iou for s in stats]
    ax.hist(final_ious, bins=bins, range=(0, 1), color="steelblue", edgecolor="white")
    ax.set_xlabel("Final IoU")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    return ax


def plot_detection_count_accuracy(
    stats: Sequence[EpisodeStats],
    dataset: Sequence,
    baseline_threshold: float = 0.5,
    baseline_nms_iou: float = 0.7,
    rl_nms_iou: float | None = None,
    detector_device=None,
    ax: plt.Axes | None = None,
    title: str = "Detection-count accuracy vs baseline",
) -> plt.Axes:
    """
    Compare how closely the RL policy and a fixed-threshold baseline match the
    ground-truth number of boxes (count accuracy in [0, 1]).
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    model = YOLO("yolov8n.pt")
    model.to(detector_device)

    episodes: List[int] = []
    rl_acc: List[float] = []
    base_acc: List[float] = []

    for s in stats:
        if s.image_idx is None or s.image_idx < 0:
            raise ValueError("EpisodeStats.image_idx missing; rerun evaluate_policy to populate it.")

        img, gt_boxes = dataset[s.image_idx]
        img_uint8 = (img * 255).clip(0, 255).astype("uint8")
        gt_count = len(gt_boxes)
        denom = max(gt_count, 1)  # avoid div-by-zero

        rl_threshold = float(s.thresholds[-1]) if s.thresholds else baseline_threshold
        rl_preds = model.predict(
            img_uint8,
            conf=rl_threshold,
            iou=float(rl_nms_iou if rl_nms_iou is not None else baseline_nms_iou),
            device=detector_device,
            verbose=False,
        )[0]
        base_preds = model.predict(
            img_uint8,
            conf=float(baseline_threshold),
            iou=float(baseline_nms_iou),
            device=detector_device,
            verbose=False,
        )[0]

        rl_count = len(rl_preds.boxes)
        base_count = len(base_preds.boxes)

        # Accuracy = 1 - normalized count error (clipped to [0, 1]).
        rl_acc.append(max(0.0, 1.0 - abs(rl_count - gt_count) / denom))
        base_acc.append(max(0.0, 1.0 - abs(base_count - gt_count) / denom))
        episodes.append(s.episode)

    ax.plot(episodes, rl_acc, marker="o", label="Policy (final threshold)")
    ax.plot(episodes, base_acc, marker="s", label=f"Baseline t={baseline_threshold:.2f}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Detection-count accuracy vs GT")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    return ax


def plot_learning_curve(
    history: Sequence[dict],
    ax: plt.Axes | None = None,
    title: str = "Learning curve (evaluation)",
) -> plt.Axes:
    """
    Plot evaluation stats (mean_final_iou / mean_return) versus timesteps.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if not history:
        ax.set_title("No evaluation history collected")
        return ax

    steps = [h.get("timesteps", 0) for h in history]
    mean_iou = [h.get("mean_final_iou", 0.0) for h in history]
    mean_return = [h.get("mean_return", 0.0) for h in history]

    ax.plot(steps, mean_iou, marker="o", label="mean_final_iou")
    ax.plot(steps, mean_return, marker="s", label="mean_return")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    return ax
