import logging
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from ultralytics import YOLO

from utility.metrics import compute_iou  # simple IoU helper
from utility.torch_utils import get_default_device

logger = logging.getLogger(__name__)

class ThresholdRefinementEnv(gym.Env):
    """
    RL environment for iterative refinement of detector post-processing knobs
    (confidence threshold, NMS IoU, and max detections).
    """
    def __init__(
        self,
        dataset,
        initial_threshold: float = 0.5,
        initial_threshold_range: tuple[float, float] | None = None,
        initial_nms_iou: float = 0.7,
        max_det: int = 200,
        device=None,
        detector_device=None,
        feature_fn: Optional[Callable] = None,
        feature_dim: int | None = None,
        topk_conf: int = 5,
        max_delta: float = 0.2,
        max_nms_delta: float = 0.1,
        max_det_delta: int = 50,
        max_steps: int = 30,
        box_count_penalty: float = 0.05,
    ):
        super().__init__()
        # PPO/policy device (used by SB3); detector can live on a separate device.
        self.device = device if device is not None else get_default_device()
        self.detector_device = (
            detector_device if detector_device is not None else self.device
        )
        self.max_delta = float(max_delta)
        self.max_nms_delta = float(max_nms_delta)
        self.max_det_delta = int(max_det_delta)
        self.box_count_penalty = float(box_count_penalty)
        self.topk_conf = max(0, int(topk_conf))
        self.model = YOLO("yolov8n.pt")  # pretrained detector
        self.model.to(self.detector_device)
        self.dataset = dataset
        self.initial_threshold = initial_threshold
        self.initial_threshold_range = initial_threshold_range
        self.current_threshold = initial_threshold
        self.initial_nms_iou = float(initial_nms_iou)
        self.current_nms_iou = float(initial_nms_iou)
        self.initial_max_det = int(max_det)
        self.current_max_det = int(max_det)
        self.max_steps = int(max_steps)
        self.feature_fn = feature_fn  # optional callable(image, preds) -> 1D np.ndarray
        self.feature_dim = feature_dim

        # If no custom feature_fn provided, default to backbone GAP embedding.
        if self.feature_fn is None:
            self._backbone = getattr(self.model.model, "model", None)
            self._detector_index = getattr(self.model.model, "detector_index", None)
            if self._backbone is not None and self._detector_index is not None:
                self.feature_dim = self._infer_backbone_dim()
                self.feature_fn = self._default_feature_fn
            else:
                self.feature_dim = 0
                self.feature_fn = None

        # Fixed action space: three controls [threshold_delta, nms_delta, max_det_delta]
        low = np.array([-self.max_delta, -self.max_nms_delta, -self.max_det_delta], dtype=np.float32)
        high = np.array([self.max_delta, self.max_nms_delta, self.max_det_delta], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        base_obs_dim = 8  # threshold, nms_iou, max_det_norm + detection stats
        total_dim = base_obs_dim + self.topk_conf + (self.feature_dim or 0)
        # Observation: [threshold, nms_iou, max_det_norm, mean_conf, std_conf, count_norm, mean_area, mean_aspect, topk_conf...] + optional features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

        self.step_count = 0
        self.prev_reward = 0.0
        self.image_idx = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        if self.initial_threshold_range:
            low, high = self.initial_threshold_range
            self.current_threshold = float(np.clip(np.random.uniform(low, high), 0.0, 1.0))
        else:
            self.current_threshold = self.initial_threshold
        self.current_nms_iou = float(np.clip(self.initial_nms_iou, 0.05, 0.99))
        self.current_max_det = max(1, self.initial_max_det)
        self.prev_reward = 0.0
        if options and "image_idx" in options and options["image_idx"] is not None:
            self.image_idx = int(np.clip(options["image_idx"], 0, len(self.dataset) - 1))
        else:
            self.image_idx = np.random.randint(0, len(self.dataset))
        logger.debug("Reset episode: idx=%d initial_threshold=%.3f", self.image_idx, self.current_threshold)
        obs = self._get_obs()
        return obs, {}

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        """
        Ultralytics expects uint8 images in [0, 255]. Our dataset provides
        float32 in [0, 1], so convert when needed.
        """

        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        return img

    def _get_obs(self):
        # Observe current threshold and mean confidence (for context)
        img, _ = self.dataset[self.image_idx]
        img_input = self._prepare_image(img)
        preds = self.model.predict(
            img_input,
            conf=self.current_threshold,
            iou=self.current_nms_iou,
            max_det=int(self.current_max_det),
            device=self.detector_device,
            verbose=False,
        )[0]

        if len(preds.boxes):
            confs = preds.boxes.conf.cpu().numpy()
            boxes = preds.boxes.xyxy.cpu().numpy()
            img_h, img_w = img_input.shape[:2]
            areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) / float(
                img_w * img_h
            )
            aspects = (boxes[:, 2] - boxes[:, 0]) / np.maximum(
                (boxes[:, 3] - boxes[:, 1]), 1e-6
            )
            mean_conf = float(np.mean(confs))
            std_conf = float(np.std(confs))
            count_norm = min(len(confs), 20) / 20.0
            mean_area = float(np.clip(np.mean(areas), 0.0, 1.0))
            mean_aspect = float(np.clip(np.mean(aspects), 0.0, 10.0) / 10.0)
        else:
            mean_conf = 0.0
            std_conf = 0.0
            count_norm = 0.0
            mean_area = 0.0
            mean_aspect = 0.0
            confs = np.array([], dtype=np.float32)

        obs_components = [
            self.current_threshold,
            self.current_nms_iou,
            float(self.current_max_det) / 500.0,  # simple normalization assuming typical caps <=500
            mean_conf,
            std_conf,
            count_norm,
            mean_area,
            mean_aspect,
        ]

        # Fixed-length top-K confidences (padded with zeros)
        if self.topk_conf > 0:
            topk = np.sort(confs)[-self.topk_conf:] if confs.size else np.array([], dtype=np.float32)
            if topk.size < self.topk_conf:
                pad = np.zeros(self.topk_conf - topk.size, dtype=np.float32)
                topk = np.concatenate([pad, topk])
            obs_components.extend(topk.tolist())

        if self.feature_fn is not None:
            feat = self.feature_fn(img_input, preds)
            feat = np.asarray(feat, dtype=np.float32).flatten()
            if self.feature_dim is not None and feat.shape[0] != self.feature_dim:
                raise ValueError(f"feature_fn returned {feat.shape[0]} dims, expected {self.feature_dim}")
            obs_components.extend(feat.tolist())

        return np.array(obs_components, dtype=np.float32)

    def _infer_backbone_dim(self) -> int:
        """
        Run a dummy forward through the backbone to infer channel dimension for pooling.
        """

        dummy = torch.zeros(1, 3, 64, 64, device=self.detector_device)
        x = dummy
        for layer in self._backbone[: self._detector_index]:
            x = layer(x)
        return int(x.shape[1])

    def _default_feature_fn(self, img_input: np.ndarray, preds):
        """
        Default feature extractor: run image through YOLO backbone and global-average-pool.
        """

        if self._backbone is None or self._detector_index is None:
            return np.zeros(self.feature_dim or 0, dtype=np.float32)

        with torch.no_grad():
            x = torch.from_numpy(img_input).to(self.detector_device)
            if x.ndim == 3:
                x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            x = x.float()
            for layer in self._backbone[: self._detector_index]:
                x = layer(x)
            pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten()
            return pooled.detach().cpu().numpy()

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32).flatten()
        threshold_delta = float(np.clip(action[0], -self.max_delta, self.max_delta))
        nms_delta = float(np.clip(action[1], -self.max_nms_delta, self.max_nms_delta)) if action.size > 1 else 0.0
        det_delta = float(np.clip(action[2], -self.max_det_delta, self.max_det_delta)) if action.size > 2 else 0.0

        self.current_threshold = np.clip(self.current_threshold + threshold_delta, 0.0, 1.0)
        self.current_nms_iou = float(np.clip(self.current_nms_iou + nms_delta, 0.05, 0.99))
        self.current_max_det = int(np.clip(self.current_max_det + det_delta, 1, 2000))

        img, gt_boxes = self.dataset[self.image_idx]
        img_input = self._prepare_image(img)
        preds = self.model.predict(
            img_input,
            conf=self.current_threshold,
            iou=self.current_nms_iou,
            max_det=int(self.current_max_det),
            device=self.detector_device,
            verbose=False,
        )[0]
        pred_boxes = preds.boxes.xyxy.cpu().numpy() if len(preds.boxes) else np.empty((0, 4))

        # Compute reward as mean IoU vs ground truth with a small penalty for box count mismatch
        reward = compute_iou(pred_boxes, gt_boxes)
        if self.box_count_penalty > 0:
            pred_count = len(pred_boxes)
            gt_count = len(gt_boxes)
            count_gap = abs(pred_count - gt_count)
            norm = max(gt_count, 1)
            reward -= self.box_count_penalty * (count_gap / norm)

        # Absolute IoU reward for a stronger learning signal
        self.prev_reward = reward

        # Only stop early for vanishing improvement after the first step
        terminated = (self.step_count > 1) and (abs(threshold_delta) < 1e-6)
        truncated = self.step_count >= self.max_steps
        obs = self._get_obs()
        logger.debug(
            "Step %d: threshold=%.3f nms_iou=%.3f max_det=%d reward=%.4f "
            "threshold_delta=%.4f nms_delta=%.4f det_delta=%.1f terminated=%s truncated=%s",
            self.step_count,
            self.current_threshold,
            self.current_nms_iou,
            int(self.current_max_det),
            reward,
            threshold_delta,
            nms_delta,
            det_delta,
            terminated,
            truncated,
        )
        return obs, reward, terminated, truncated, {}
