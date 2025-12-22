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
    (confidence threshold and NMS IoU).
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
        max_steps: int = 30,
        reward_scale: float = 1.0,
        duplicate_penalty: float = 0.05,
        duplicate_iou_threshold: float = 0.8,
        box_count_penalty: float = 0.05,
        fbeta: float = 2.0,
        fn_penalty: float = 0.0,
    ):
        super().__init__()
        # PPO/policy device (used by SB3); detector can live on a separate device.
        self.device = device if device is not None else get_default_device()
        self.detector_device = (
            detector_device if detector_device is not None else self.device
        )
        self.max_delta = float(max_delta)
        self.max_nms_delta = float(max_nms_delta)
        self.duplicate_penalty = float(duplicate_penalty)
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
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
        self.reward_scale = float(reward_scale)
        self.fbeta = float(fbeta)
        self.fn_penalty = float(fn_penalty)

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

        # Fixed action space: two controls [threshold_delta, nms_delta]
        low = np.array([-self.max_delta, -self.max_nms_delta], dtype=np.float32)
        high = np.array([self.max_delta, self.max_nms_delta], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        base_obs_dim = 7  # threshold, nms_iou + detection stats
        total_dim = base_obs_dim + self.topk_conf + (self.feature_dim or 0)
        # Observation: [threshold, nms_iou, mean_conf, std_conf, count_norm, mean_area, mean_aspect, topk_conf...] + optional features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)

        self.step_count = 0
        self.prev_reward = 0.0
        self.last_precision = 0.0
        self.last_recall = 0.0
        self.last_f1 = 0.0
        self.last_matched_iou = 0.0
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
        self.last_precision = 0.0
        self.last_recall = 0.0
        self.last_f1 = 0.0
        self.last_matched_iou = 0.0
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

    def _pairwise_iou(self, boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
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
        self,
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

        ious = self._pairwise_iou(pred_boxes, gt_boxes)
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

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32).flatten()
        threshold_delta = float(np.clip(action[0], -self.max_delta, self.max_delta))
        nms_delta = float(np.clip(action[1], -self.max_nms_delta, self.max_nms_delta)) if action.size > 1 else 0.0

        self.current_threshold = np.clip(self.current_threshold + threshold_delta, 0.0, 1.0)
        self.current_nms_iou = float(np.clip(self.current_nms_iou + nms_delta, 0.05, 0.99))

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

        # Compute reward using matched IoU * F-beta to balance accuracy and count (beta>1 favors recall).
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            mean_iou = 1.0
            precision = 1.0
            recall = 1.0
            fbeta = 1.0
            fn = 0
            gt_count = 0
        else:
            matched_ious, tp, fp, fn = self._match_boxes(
                pred_boxes,
                gt_boxes,
                iou_threshold=float(self.current_nms_iou),
            )
            mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            beta2 = self.fbeta ** 2
            denom = (beta2 * precision + recall)
            fbeta = ((1.0 + beta2) * precision * recall / denom) if denom > 0 else 0.0
            gt_count = len(gt_boxes)

        reward = mean_iou * fbeta
        if self.fn_penalty > 0 and gt_count > 0:
            reward -= self.fn_penalty * (fn / gt_count)
        if self.box_count_penalty > 0:
            pred_count = len(pred_boxes)
            count_gap = abs(pred_count - gt_count)
            norm = max(gt_count, 1)
            reward -= self.box_count_penalty * (count_gap / norm)
        if self.duplicate_penalty > 0 and len(pred_boxes) > 1:
            dup_pairs = 0
            for i in range(len(pred_boxes)):
                for j in range(i + 1, len(pred_boxes)):
                    iou_pair = compute_iou(pred_boxes[i : i + 1], pred_boxes[j : j + 1])
                    if iou_pair >= self.duplicate_iou_threshold:
                        dup_pairs += 1
            if dup_pairs:
                reward -= self.duplicate_penalty * dup_pairs

        reward *= self.reward_scale
        self.last_precision = float(precision)
        self.last_recall = float(recall)
        self.last_f1 = float(fbeta)
        self.last_matched_iou = float(mean_iou)
        # Absolute IoU reward for a stronger learning signal
        self.prev_reward = reward

        # Only stop early for vanishing improvement after the first step
        terminated = (self.step_count > 1) and (abs(threshold_delta) < 1e-6)
        truncated = self.step_count >= self.max_steps
        obs = self._get_obs()
        logger.debug(
            "Step %d: threshold=%.3f nms_iou=%.3f max_det=%d reward=%.4f "
            "threshold_delta=%.4f nms_delta=%.4f terminated=%s truncated=%s",
            self.step_count,
            self.current_threshold,
            self.current_nms_iou,
            int(self.current_max_det),
            reward,
            threshold_delta,
            nms_delta,
            terminated,
            truncated,
        )
        return obs, reward, terminated, truncated, {}
