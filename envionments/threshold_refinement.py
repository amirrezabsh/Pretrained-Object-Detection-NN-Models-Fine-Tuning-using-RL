import gym
import numpy as np
from gym import spaces
from ultralytics import YOLO
from utility.metrics import compute_iou  # simple IoU helper

class ThresholdRefinementEnv(gym.Env):
    """
    RL environment for iterative refinement of YOLO detection threshold.
    """
    def __init__(self, dataset, initial_threshold=0.5):
        super().__init__()
        self.model = YOLO("yolov8n.pt")  # pretrained detector
        self.dataset = dataset
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.max_steps = 10

        # Continuous action: change threshold in [-0.1, +0.1]
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)

        # Observation: [current threshold, mean_confidence_of_detections]
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        self.step_count = 0
        self.prev_reward = 0.0
        self.image_idx = 0

    def reset(self):
        self.step_count = 0
        self.current_threshold = self.initial_threshold
        self.prev_reward = 0.0
        self.image_idx = np.random.randint(0, len(self.dataset))
        return self._get_obs()

    def _get_obs(self):
        # Observe current threshold and mean confidence (for context)
        img, _ = self.dataset[self.image_idx]
        preds = self.model.predict(img, conf=self.current_threshold, verbose=False)[0]
        mean_conf = preds.boxes.conf.cpu().numpy().mean() if len(preds.boxes) else 0.0
        return np.array([self.current_threshold, mean_conf], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        self.current_threshold = np.clip(self.current_threshold + float(action[0]), 0.0, 1.0)

        img, gt_boxes = self.dataset[self.image_idx]
        preds = self.model.predict(img, conf=self.current_threshold, verbose=False)[0]
        pred_boxes = preds.boxes.xyxy.cpu().numpy() if len(preds.boxes) else np.empty((0, 4))

        # Compute reward as mean IoU vs ground truth
        reward = compute_iou(pred_boxes, gt_boxes)

        # Optional: reward improvement instead of absolute IoU
        delta = reward - self.prev_reward
        self.prev_reward = reward

        done = (self.step_count >= self.max_steps) or (abs(delta) < 1e-4)
        obs = self._get_obs()
        return obs, delta, done, {}

