from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from envionments.threshold_refinement import ThresholdRefinementEnv


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


def evaluate_policy(
    model: BaseAlgorithm,
    dataset: Sequence,
    episodes: int = 5,
    deterministic: bool = True,
) -> List[EpisodeStats]:
    """
    Roll out a trained RL policy and collect summary statistics.
    """

    env = ThresholdRefinementEnv(dataset)
    stats: List[EpisodeStats] = []

    for ep in range(episodes):
        obs = env.reset()
        thresholds = [float(obs[0])]
        done = False
        rewards = []
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            thresholds.append(float(obs[0]))
            rewards.append(float(reward))
            step += 1

        stats.append(
            EpisodeStats(
                episode=ep,
                steps=step,
                final_iou=float(env.prev_reward),
                reward_sum=float(np.sum(rewards)),
                thresholds=thresholds,
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

    return {
        "episodes": len(final_ious),
        "mean_final_iou": float(np.mean(final_ious)),
        "std_final_iou": float(np.std(final_ious)),
        "mean_return": float(np.mean(reward_sums)),
        "mean_steps": float(np.mean(steps)),
    }


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
