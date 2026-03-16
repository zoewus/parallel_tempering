# config.py

import torch

# device / general
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# diffusion
N_DIFFUSION_STEPS = 100

# training hyperparameters
TRAINING = {
    "n_steps": 500_000,
    "batch_size": 512,
    "lr": 2e-4
}

# datasets
DATASETS = {
    "single":   {"n_samples": 50_000, "means": [0.0],        "stds": 1.0, 		"step_scale": 0.1, 		"n_langevin_steps":3},
    "barrier":  {"n_samples": 50_000, "means": [-3.0, 3.0],  "stds": 0.5, 		"step_scale": 0.1, 		"n_langevin_steps":3},
    "composed": {"n_samples": 100_000, "means": [-3.0, 0.0, 3.0], "stds": [0.5, 1.0, 0.5], 		"step_scale": 0.1, 		"n_langevin_steps":3},
	"single_overlapping":   {"n_samples": 25_000, "means": [0.0],        "stds": 0.5, 		"step_scale": 0.05, 		"n_langevin_steps":5},
    "barrier_overlapping":  {"n_samples": 50_000, "means": [-3.0, 3.0],  "stds": 2.0, 		"step_scale": 0.05, 		"n_langevin_steps":5},
    "composed_overlapping": {"n_samples": 75_000, "means": [-3.0, 0.0, 3.0], "stds": [2.0, 0.5, 2.0], 		"step_scale": 0.05, 		"n_langevin_steps":5},
}

CKPT_DIR = "model_checkpoints"
