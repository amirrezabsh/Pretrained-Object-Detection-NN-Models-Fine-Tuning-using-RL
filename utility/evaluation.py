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
    nms_ious: List[float]
    ious: List[float]
    rewards: List[float]
    max_det: int | None = None
    final_precision: float | None = None
    final_recall: float | None = None
    final_f1: float | None = None
    final_matched_iou: float | None = None
    image_idx: int | None = None


def _select_stats(stats: Sequence[EpisodeStats], max_episodes: int | None) -> Sequence[EpisodeStats]:
    """
    Keep at most `max_episodes` items, preferring highest final IoU episodes to
    make busy plots more readable.
    """

    if max_episodes is None or len(stats) <= max_episodes:
        return stats
    return sorted(stats, key=lambda s: s.final_iou, reverse=True)[:max_episodes]


def _mean_std_over_steps(sequences: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean/std across ragged sequences by padding with NaN.
    """

    if not sequences:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    max_len = max(len(seq) for seq in sequences)
    arr = np.full((len(sequences), max_len), np.nan, dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq_np = np.asarray(seq, dtype=np.float32)
        arr[i, : len(seq_np)] = seq_np
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    steps = np.arange(max_len)
    return mean, std, steps


def evaluate_policy(
    model: BaseAlgorithm,
    dataset: Sequence,
    episodes: int = 5,
    deterministic: bool = True,
    device=None,
    detector_device=None,
    env_kwargs: dict | None = None,
    sample_indices: Sequence[int] | None = None,
) -> List[EpisodeStats]:
    """
    Roll out a trained RL policy and collect summary statistics.

    Parameters
    ----------
    sample_indices : Sequence[int] | None
        Explicit image indices to evaluate. When None (default), the evaluator
        samples without replacement when possible to avoid reusing the same
        image across episodes, then falls back to random sampling with
        replacement if `episodes` exceeds the dataset size.
    """

    env_kwargs = env_kwargs or {}
    env = ThresholdRefinementEnv(dataset, device=device, detector_device=detector_device, **env_kwargs)
    stats: List[EpisodeStats] = []
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; cannot run evaluation.")

    if sample_indices is not None:
        episode_indices = list(sample_indices)
    else:
        # Prefer sampling without replacement to avoid repeating the same image across episodes.
        if len(dataset) >= episodes:
            episode_indices = np.random.permutation(len(dataset)).tolist()[:episodes]
        else:
            # Cover every sample once, then allow repeats if more episodes are requested than data size.
            base = np.random.permutation(len(dataset)).tolist()
            extra = np.random.randint(0, len(dataset), size=episodes - len(base)).tolist()
            episode_indices = base + extra

    for ep in range(episodes):
        idx = episode_indices[ep % len(episode_indices)]
        obs, _ = env.reset(options={"image_idx": idx})
        thresholds = [float(obs[0])]
        nms_ious = [float(getattr(env, "current_nms_iou", 0.0))]
        max_det = int(getattr(env, "current_max_det", 0)) if hasattr(env, "current_max_det") else None
        done = False
        rewards = []
        ious = [float(env.prev_reward)]
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            thresholds.append(float(obs[0]))
            nms_ious.append(float(getattr(env, "current_nms_iou", nms_ious[-1])))
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
                nms_ious=nms_ious,
                ious=ious,
                rewards=rewards,
                max_det=int(max_det) if max_det is not None else None,
                final_precision=float(getattr(env, "last_precision", 0.0)),
                final_recall=float(getattr(env, "last_recall", 0.0)),
                final_f1=float(getattr(env, "last_f1", 0.0)),
                final_matched_iou=float(getattr(env, "last_matched_iou", 0.0)),
                image_idx=int(idx),
            )
        )

    return stats


def summarize_stats(stats: Iterable[EpisodeStats]) -> dict:
    """
    Aggregate statistics (mean/std) for quick notebook display.
    """

    def _mean_std(values: Iterable[float | None]) -> tuple[float, float]:
        filtered = [v for v in values if v is not None]
        if not filtered:
            return 0.0, 0.0
        arr = np.array(filtered, dtype=np.float32)
        return float(np.mean(arr)), float(np.std(arr))

    final_ious = np.array([s.final_iou for s in stats], dtype=np.float32)
    final_thresholds = np.array([s.thresholds[-1] for s in stats], dtype=np.float32)
    final_nms = np.array([s.nms_ious[-1] for s in stats], dtype=np.float32)
    reward_sums = np.array([s.reward_sum for s in stats], dtype=np.float32)
    steps = np.array([s.steps for s in stats], dtype=np.float32)
    mean_final_precision, std_final_precision = _mean_std(s.final_precision for s in stats)
    mean_final_recall, std_final_recall = _mean_std(s.final_recall for s in stats)
    mean_final_f1, std_final_f1 = _mean_std(s.final_f1 for s in stats)
    mean_final_matched_iou, std_final_matched_iou = _mean_std(s.final_matched_iou for s in stats)

    summary = {
        "episodes": len(final_ious),
        "mean_final_iou": float(np.mean(final_ious)),
        "std_final_iou": float(np.std(final_ious)),
        "mean_return": float(np.mean(reward_sums)),
        "mean_steps": float(np.mean(steps)),
        "mean_final_threshold": float(np.mean(final_thresholds)),
        "std_final_threshold": float(np.std(final_thresholds)),
        "mean_final_nms_iou": float(np.mean(final_nms)),
        "std_final_nms_iou": float(np.std(final_nms)),
        "mean_final_precision": mean_final_precision,
        "std_final_precision": std_final_precision,
        "mean_final_recall": mean_final_recall,
        "std_final_recall": std_final_recall,
        "mean_final_f1": mean_final_f1,
        "std_final_f1": std_final_f1,
        "mean_final_matched_iou": mean_final_matched_iou,
        "std_final_matched_iou": std_final_matched_iou,
    }
    logger.info(
        "Evaluation summary: episodes=%d mean_final_iou=%.4f std_final_iou=%.4f "
        "mean_return=%.4f mean_steps=%.2f mean_final_threshold=%.3f mean_final_nms=%.3f",
        summary["episodes"],
        summary["mean_final_iou"],
        summary["std_final_iou"],
        summary["mean_return"],
        summary["mean_steps"],
        summary["mean_final_threshold"],
        summary["mean_final_nms_iou"],
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
    max_episodes: int | None = None,
    aggregate: bool = False,
    show_std_band: bool = True,
) -> plt.Axes:
    """
    Visualize how the policy adapts the threshold over timesteps.

    Parameters
    ----------
    max_episodes : int | None
        When set, plot only the top-N episodes by final IoU to reduce clutter.
    aggregate : bool
        When True, plot mean +/- std across all episodes instead of individual
        trajectories (helpful when there are many episodes).
    show_std_band : bool
        Whether to draw the std dev band when aggregate=True.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if aggregate:
        if not stats:
            return ax
        mean, std, steps = _mean_std_over_steps([s.thresholds for s in stats])
        ax.plot(steps, mean, color="tab:blue", label="mean threshold")
        if show_std_band:
            ax.fill_between(steps, mean - std, mean + std, color="tab:blue", alpha=0.2, label="std band")
    else:
        for s in _select_stats(stats, max_episodes):
            ax.plot(
                range(len(s.thresholds)),
                s.thresholds,
                marker="o",
                label=f"ep {s.episode} (t={s.thresholds[-1]:.2f}, nms={s.nms_ious[-1]:.2f})",
            )

    ax.set_xlabel("Step")
    ax.set_ylabel("Confidence threshold")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    return ax


def plot_nms_trajectories(
    stats: Sequence[EpisodeStats],
    ax: plt.Axes | None = None,
    title: str = "NMS IoU trajectory per evaluation episode",
    max_episodes: int | None = None,
    aggregate: bool = False,
    show_std_band: bool = True,
) -> plt.Axes:
    """
    Visualize how NMS IoU is adjusted over timesteps.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if aggregate:
        if not stats:
            return ax
        mean, std, steps = _mean_std_over_steps([s.nms_ious for s in stats])
        ax.plot(steps, mean, color="tab:green", label="mean NMS IoU")
        if show_std_band:
            ax.fill_between(steps, mean - std, mean + std, color="tab:green", alpha=0.2, label="std band")
    else:
        for s in _select_stats(stats, max_episodes):
            ax.plot(
                range(len(s.nms_ious)),
                s.nms_ious,
                marker="o",
                label=f"ep {s.episode} (final nms={s.nms_ious[-1]:.2f})",
            )

    ax.set_xlabel("Step")
    ax.set_ylabel("NMS IoU")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)
    return ax


def plot_iou_trajectories(
    stats: Sequence[EpisodeStats],
    ax: plt.Axes | None = None,
    title: str = "IoU trajectory per evaluation episode",
    max_episodes: int | None = None,
    aggregate: bool = False,
    show_std_band: bool = True,
) -> plt.Axes:
    """
    Visualize IoU progression over steps (absolute IoU, not deltas).
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if aggregate:
        if not stats:
            return ax
        mean, std, steps = _mean_std_over_steps([s.ious for s in stats])
        ax.plot(steps, mean, color="tab:red", label="mean IoU")
        if show_std_band:
            ax.fill_between(steps, mean - std, mean + std, color="tab:red", alpha=0.2, label="std band")
    else:
        for s in _select_stats(stats, max_episodes):
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
    max_episodes: int | None = None,
    aggregate: bool = False,
    show_std_band: bool = True,
) -> plt.Axes:
    """
    Plot reward deltas per step to see when improvements happen.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if aggregate:
        if not stats:
            return ax
        mean, std, steps = _mean_std_over_steps([s.rewards for s in stats])
        ax.plot(steps, mean, color="tab:purple", label="mean reward")
        if show_std_band:
            ax.fill_between(steps, mean - std, mean + std, color="tab:purple", alpha=0.2, label="std band")
    else:
        for s in _select_stats(stats, max_episodes):
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
    baseline_max_det: int = 200,
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
            max_det=int(baseline_max_det),
            device=detector_device,
            verbose=False,
        )[0]
        base_boxes = base_preds.boxes.xyxy.cpu().numpy() if len(base_preds.boxes) else np.empty((0, 4))

        # Plot RL result
        ax_rl = axes[row, 0]
        ax_rl.imshow(img_uint8)
        _draw_boxes(
            ax_rl,
            rl_boxes,
            color="tab:green",
            label=f"Policy t={env.current_threshold:.2f} nms={getattr(env, 'current_nms_iou', baseline_nms_iou):.2f} max={int(getattr(env, 'current_max_det', 200))}",
            y_offset=0.99,
        )
        _draw_boxes(ax_rl, gt_boxes, color="tab:orange", label="GT", y_offset=0.87)
        ax_rl.axis("off")
        ax_rl.set_title(
            f"Policy (idx {idx}) IoU={compute_iou(rl_boxes, gt_boxes):.3f} "
            f"dets={len(rl_boxes)} | GT={len(gt_boxes)}"
        )

        # Plot baseline result
        ax_base = axes[row, 1]
        ax_base.imshow(img_uint8)
        _draw_boxes(
            ax_base,
            base_boxes,
            color="tab:blue",
            label=f"Baseline t={baseline_threshold:.2f} nms={baseline_nms_iou:.2f} max={int(baseline_max_det)}",
            y_offset=0.99,
        )
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


def _pairwise_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of boxes (xyxy).
    """

    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    ious = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes_a):
        for j, (gx1, gy1, gx2, gy2) in enumerate(boxes_b):
            inter_x1 = max(x1, gx1)
            inter_y1 = max(y1, gy1)
            inter_x2 = min(x2, gx2)
            inter_y2 = min(y2, gy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union_area = (x2 - x1) * (y2 - y1) + (gx2 - gx1) * (gy2 - gy1) - inter_area
            if union_area > 0:
                ious[i, j] = inter_area / union_area
    return ious


def _match_boxes(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> tuple[list[float], int, int, int]:
    """
    Greedy matching between predictions and GT using IoU threshold.
    Returns matched IoUs plus tp/fp/fn counts.
    """

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], 0, len(pred_boxes), len(gt_boxes)

    ious = _pairwise_iou(pred_boxes, gt_boxes)
    matched_ious: list[float] = []
    while True:
        max_idx = np.unravel_index(int(np.argmax(ious)), ious.shape)
        max_iou = float(ious[max_idx])
        if max_iou < iou_threshold:
            break
        matched_ious.append(max_iou)
        ious[max_idx[0], :] = -1.0
        ious[:, max_idx[1]] = -1.0

    tp = len(matched_ious)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return matched_ious, tp, fp, fn


def evaluate_baseline_detection_metrics(
    stats: Sequence[EpisodeStats],
    dataset: Sequence,
    baseline_threshold: float = 0.5,
    baseline_nms_iou: float = 0.7,
    baseline_max_det: int = 200,
    detector_device=None,
) -> dict:
    """
    Evaluate baseline detector metrics on the same images used in `stats`.
    """

    model = YOLO("yolov8n.pt")
    model.to(detector_device)

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    matched_ious: List[float] = []

    for s in stats:
        if s.image_idx is None or s.image_idx < 0:
            raise ValueError("EpisodeStats.image_idx missing; rerun evaluate_policy to populate it.")

        img, gt_boxes = dataset[s.image_idx]
        img_uint8 = (img * 255).clip(0, 255).astype("uint8")
        preds = model.predict(
            img_uint8,
            conf=float(baseline_threshold),
            iou=float(baseline_nms_iou),
            max_det=int(baseline_max_det),
            device=detector_device,
            verbose=False,
        )[0]
        pred_boxes = preds.boxes.xyxy.cpu().numpy() if len(preds.boxes) else np.empty((0, 4))
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            matched_ious.append(1.0)
            continue

        matched, tp, fp, fn = _match_boxes(
            pred_boxes,
            gt_boxes,
            iou_threshold=float(baseline_nms_iou),
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        mean_iou = float(np.mean(matched)) if matched else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        matched_ious.append(mean_iou)

    return {
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def compute_detection_count_accuracy(
    stats: Sequence[EpisodeStats],
    dataset: Sequence,
    baseline_threshold: float = 0.5,
    baseline_nms_iou: float = 0.7,
    baseline_max_det: int = 200,
    rl_nms_iou: float | None = None,
    rl_max_det: int | None = None,
    detector_device=None,
) -> dict:
    """
    Compute detection-count accuracy for policy vs baseline on the same images.
    """

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
        # Prefer per-episode settings if provided
        rl_nms = float(rl_nms_iou if rl_nms_iou is not None else (s.nms_ious[-1] if s.nms_ious else baseline_nms_iou))
        rl_max = int(
            rl_max_det
            if rl_max_det is not None
            else (s.max_det if s.max_det is not None else baseline_max_det)
        )
        rl_preds = model.predict(
            img_uint8,
            conf=rl_threshold,
            iou=rl_nms,
            max_det=rl_max,
            device=detector_device,
            verbose=False,
        )[0]
        base_preds = model.predict(
            img_uint8,
            conf=float(baseline_threshold),
            iou=float(baseline_nms_iou),
            max_det=int(baseline_max_det),
            device=detector_device,
            verbose=False,
        )[0]

        rl_count = len(rl_preds.boxes)
        base_count = len(base_preds.boxes)

        # Accuracy = 1 - normalized count error (clipped to [0, 1]).
        rl_acc.append(max(0.0, 1.0 - abs(rl_count - gt_count) / denom))
        base_acc.append(max(0.0, 1.0 - abs(base_count - gt_count) / denom))
        episodes.append(s.episode)

    return {"episodes": episodes, "rl_acc": rl_acc, "baseline_acc": base_acc}


def plot_detection_count_accuracy(
    stats: Sequence[EpisodeStats],
    dataset: Sequence,
    baseline_threshold: float = 0.5,
    baseline_nms_iou: float = 0.7,
    baseline_max_det: int = 200,
    rl_nms_iou: float | None = None,
    rl_max_det: int | None = None,
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

    results = compute_detection_count_accuracy(
        stats=stats,
        dataset=dataset,
        baseline_threshold=baseline_threshold,
        baseline_nms_iou=baseline_nms_iou,
        baseline_max_det=baseline_max_det,
        rl_nms_iou=rl_nms_iou,
        rl_max_det=rl_max_det,
        detector_device=detector_device,
    )

    ax.plot(results["episodes"], results["rl_acc"], marker="o", label="Policy (final threshold)")
    ax.plot(results["episodes"], results["baseline_acc"], marker="s", label=f"Baseline t={baseline_threshold:.2f}")
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
