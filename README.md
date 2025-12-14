# Pretrained Object Detection NN Models Fine-Tuning using RL

This project explores how to fine-tune the post-processing decisions of pretrained
object detectors via reinforcement learning. Instead of retraining YOLO weights,
we train an RL agent (Stable-Baselines3 PPO) to adjust post-processing knobs
(confidence threshold, NMS IoU, and max detections) on top of a frozen detector,
maximizing IoU with ground-truth boxes.

## Repository Layout

- `envionments/threshold_refinement.py` – Gymnasium environment that wraps a detector;
  actions jointly adjust confidence threshold, NMS IoU, and max detections. Rewards
  are IoU deltas with a small box-count penalty.
- `utility/metrics.py` – IoU helper shared between the environment and evaluation.
- `utility/dataset.py` – Loaders for YOLO-format datasets plus a torchvision-backed
  Pascal VOC 2007 adapter with optional auto-download support.
- `utility/evaluation.py` – Helpers to evaluate trained policies, compute summary
  stats, and visualize trajectories (threshold, IoU, reward) plus detection-count
  accuracy versus a fixed baseline.
- `main.ipynb` – End-to-end notebook for installing dependencies, loading data,
  training PPO, and evaluating the learned policy.
- `.env` / `.env.example` - Environment variable configuration (dataset paths,
  RL hyperparameters).

## Comparing RL vs baseline YOLO

After training a PPO policy, you can compare it to the frozen YOLOv8 baseline:

```bash
python -m utility.model_comparison --model path/to/ppo_model.zip
```

This will:
- Load the dataset from `.env` (custom YOLO-format if `IMAGE_DIR`/`LABEL_DIR` are set,
  otherwise Pascal VOC 2007 with auto-download).
- Evaluate the RL policy for `RL_EVAL_EPISODES` rollouts.
- Sweep the baseline YOLO across fixed thresholds (0.3-0.7) and report the best mean IoU.

## Quick Start

1. **Create a virtual environment** (recommended) and install dependencies from
   the notebook’s first cell or manually:
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure environment variables**:
   - Copy `.env.example` to `.env`.
   - Adjust values (dataset paths, training steps, etc.). Example:
     ```
     VOC_ROOT=./data/voc
     RL_TOTAL_TIMESTEPS=20000
     RL_DATA_LIMIT=500
     ```
3. **Provide data**:
   - Default behavior uses Pascal VOC 2007 (`load_pascal_voc2007`). If the dataset
     isn’t present, it will be downloaded automatically beneath `VOC_ROOT`.
   - To use a custom YOLO-format dataset, set `IMAGE_DIR` and `LABEL_DIR` in `.env`.
4. **Run the notebook**:
   - Execute the `%pip install ...` cell to ensure packages are installed in the
     active kernel.
   - Run the training cell; PPO will learn to adjust thresholds based on your data.
   - Run the evaluation cell to print summary metrics and plot threshold trajectories.

## Environment Variables

| Variable             | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| `VOC_ROOT`           | Root directory for Pascal VOC downloads (default `./data/voc`).    |
| `IMAGE_DIR`          | Path to custom dataset images (optional).                          |
| `LABEL_DIR`          | Path to custom YOLO TXT labels (optional).                         |
| `RL_DATA_LIMIT`      | Number of samples to load for quicker experimentation.             |
| `RL_TOTAL_TIMESTEPS` | PPO training timesteps.                                            |
| `RL_LEARNING_RATE`   | PPO learning rate.                                                 |
| `RL_EVAL_EPISODES`   | Number of evaluation rollouts to average over.                     |
| `RL_MAX_DELTA`       | Max per-step change to the confidence threshold.                   |
| `RL_MAX_STEPS`       | Steps per episode for threshold/NMS/max-det refinement.            |
| `RL_NMS_IOU`         | Initial NMS IoU for the detector.                                 |
| `RL_MAX_NMS_DELTA`   | Max per-step change to NMS IoU.                                    |
| `RL_MAX_DET`         | Initial max detections for the detector.                           |
| `RL_MAX_DET_DELTA`   | Max per-step change to max detections.                             |
| `RL_BOX_COUNT_PENALTY` | Penalty weight for box count mismatch in the reward.             |

If `IMAGE_DIR`/`LABEL_DIR` are provided, the loader switches from VOC to the
custom dataset automatically.

## Notes

- `.gitignore` excludes `.env`, datasets (`data/`), checkpoints, and virtual envs.
- The project now tunes multiple model-agnostic post-processing parameters
  (threshold, NMS IoU, max detections). Adapters can be added to wrap other
  detectors while keeping the same RL loop.

### macOS Metal / MPS

- PyTorch on Apple Silicon works best with an MPS build (standard pip wheels now include it). You can quickly check availability in a Python shell:
  ```python
  import torch
  print(torch.backends.mps.is_available(), torch.backends.mps.is_built())
  ```
- The code automatically prefers `mps` -> `cuda` -> `cpu`. To force a device, export `TORCH_DEVICE` (e.g., `TORCH_DEVICE=cpu`).
- Both the YOLO baseline and PPO policy load onto the selected device, and Ultralytics predictions explicitly run there.

Feel free to adapt the notebook into Python scripts if you need automation or CI.
