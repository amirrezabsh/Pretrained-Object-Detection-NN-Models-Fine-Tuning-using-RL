from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from envionments.threshold_refinement import ThresholdRefinementEnv

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


def evaluate_policy(
    model: BaseAlgorithm,
    dataset: Sequence,
    episodes: int = 5,
    deterministic: bool = True,
    device=None,
    detector_device=None,
) -> List[EpisodeStats]:
    """
    Roll out a trained RL policy and collect summary statistics.
    """

    env = ThresholdRefinementEnv(dataset, device=device, detector_device=detector_device)
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
