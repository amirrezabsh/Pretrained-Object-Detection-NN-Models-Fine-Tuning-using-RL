import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ultralytics import YOLO

from utility.metrics import compute_iou  # simple IoU helper
from utility.torch_utils import get_default_device

logger = logging.getLogger(__name__)

class ThresholdRefinementEnv(gym.Env):
    """
    RL environment for iterative refinement of YOLO detection threshold.
    """
    def __init__(self, dataset, initial_threshold=0.5, device=None, detector_device=None):
        super().__init__()
        # PPO/policy device (used by SB3); detector can live on a separate device.
        self.device = device if device is not None else get_default_device()
        self.detector_device = (
            detector_device if detector_device is not None else self.device
        )
        self.model = YOLO("yolov8n.pt")  # pretrained detector
        self.model.to(self.detector_device)
        self.dataset = dataset
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.max_steps = 10

        # Continuous action: change threshold in [-0.1, +0.1]
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)

        # Observation: [threshold, mean_conf, std_conf, count_norm, mean_area, mean_aspect]
        # count_norm caps detection count at 20 and scales to [0,1]; areas are relative to image area.
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

        self.step_count = 0
        self.prev_reward = 0.0
        self.image_idx = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_threshold = self.initial_threshold
        self.prev_reward = 0.0
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

        return np.array(
            [
                self.current_threshold,
                mean_conf,
                std_conf,
                count_norm,
                mean_area,
                mean_aspect,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        self.step_count += 1
        self.current_threshold = np.clip(self.current_threshold + float(action[0]), 0.0, 1.0)

        img, gt_boxes = self.dataset[self.image_idx]
        img_input = self._prepare_image(img)
        preds = self.model.predict(
            img_input,
            conf=self.current_threshold,
            device=self.detector_device,
            verbose=False,
        )[0]
        pred_boxes = preds.boxes.xyxy.cpu().numpy() if len(preds.boxes) else np.empty((0, 4))

        # Compute reward as mean IoU vs ground truth
        reward = compute_iou(pred_boxes, gt_boxes)

        # Optional: reward improvement instead of absolute IoU
        delta = reward - self.prev_reward
        self.prev_reward = reward

        # Only stop early for vanishing improvement after the first step
        terminated = (self.step_count > 1) and (abs(delta) < 1e-4)
        truncated = self.step_count >= self.max_steps
        obs = self._get_obs()
        logger.debug(
            "Step %d: threshold=%.3f reward=%.4f delta=%.4f terminated=%s truncated=%s",
            self.step_count,
            self.current_threshold,
            reward,
            delta,
            terminated,
            truncated,
        )
        return obs, delta, terminated, truncated, {}
