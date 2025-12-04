# Pretrained Object Detection NN Models Fine-Tuning using RL

This project explores how to fine-tune the post-processing decisions of pretrained
object detectors via reinforcement learning. Instead of retraining YOLO weights,
we train an RL agent (Stable-Baselines3 PPO) to adjust the confidence threshold on
top of a frozen YOLOv8 model, maximizing IoU with ground-truth boxes.

## Repository Layout

- `envionments/threshold_refinement.py` – Gymnasium environment that wraps a YOLO
  detector; actions adjust the confidence threshold and rewards are IoU deltas.
- `utility/metrics.py` – IoU helper shared between the environment and evaluation.
- `utility/dataset.py` – Loaders for YOLO-format datasets plus a torchvision-backed
  Pascal VOC 2007 adapter with optional auto-download support.
- `utility/evaluation.py` – Helpers to evaluate trained policies, compute summary
  stats, and visualize threshold trajectories.
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

If `IMAGE_DIR`/`LABEL_DIR` are provided, the loader switches from VOC to the
custom dataset automatically.

## Notes

- `.gitignore` excludes `.env`, datasets (`data/`), checkpoints, and virtual envs.
- The project currently focuses on confidence-threshold tuning, but the environment
  structure allows experimentation with other post-processing parameters or reward
  formulations.

Feel free to adapt the notebook into Python scripts if you need automation or CI.
