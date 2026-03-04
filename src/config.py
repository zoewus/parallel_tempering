# config.py

import torch

# device / general
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# diffusion
N_DIFFUSION_STEPS = 100

# training hyperparameters
TRAINING = {
    "n_steps": 200_000,
    "batch_size": 512,
    "lr": 2e-4
}

# sampling hyperparameters
SAMPLING = {
    "n_langevin_steps": 10,
    "n_samples": 50_000,
}

# datasets
DATASETS = {
    "single":   {"n_samples": 50_000, "means": [0.0],        "stds": 1.0},
    "barrier":  {"n_samples": 50_000, "means": [-3.0, 3.0],  "stds": 0.5},
    "composed": {"n_samples": 100_000, "means": [-3.0, 0.0, 3.0], "stds": [0.5, 0.5, 0.5]},
}

CKPT_DIR = "model_checkpoints"
