import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .model import MLP       
from .dataset import generate_gaussian_mixture
from .schedule import betas, alphas, alpha_bars, ts_desc
from .config import DEVICE, TRAINING, DATASETS, N_DIFFUSION_STEPS, CKPT_DIR

device = DEVICE
n_steps = TRAINING["n_steps"]
batch_size = TRAINING["batch_size"]
lr = TRAINING["lr"]
ckpt_dir = CKPT_DIR
n_diffusion_steps = N_DIFFUSION_STEPS

torch.manual_seed(42)     

# ---------- training function (train_model) ----------
def train_model(config, temperature=1.0, log_every=5000):
	"""Trains model according to a dataset defined by dataset_config"""
	name, dataset_config = config

	x0_all = generate_gaussian_mixture(
        n_samples=dataset_config["n_samples"],
		means=dataset_config["means"],
		stds=dataset_config["stds"],
        device='cpu',
    )

	loader = DataLoader(
		TensorDataset(x0_all),
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=0,      # keep simple; set >0 if you want
		pin_memory=True,    # helps H2D transfer
	)

	model = MLP().to(device)
	opt = optim.Adam(model.parameters(), lr=lr)
	model.train()

	data_iter = iter(loader)

	for step in range(1, n_steps + 1):
		try:
			(x0,) = next(data_iter)
		except StopIteration:
			data_iter = iter(loader)  # reshuffles because shuffle=True
			(x0,) = next(data_iter)

		x0 = x0.to(device, non_blocking=True)

		t = torch.randint(0, n_diffusion_steps, (batch_size, 1), device=device)
		a_bar = alpha_bars[t]
		
		noise = torch.randn_like(x0)
		xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

		eps_hat = model(xt, t)
		loss = ((noise - eps_hat) ** 2).mean()

		opt.zero_grad()
		loss.backward()
		opt.step()

		if step % log_every == 0:
			print(f"temperature={temperature} step={step} loss={loss.item():.4f}")

	save_path = f"{ckpt_dir}/{name}_{temperature:.1f}.pt"
	torch.save(model.state_dict(), save_path)
	print(f"Trained model saved to {save_path}")

	return model


if __name__ == "main":
	for dataset_config in DATASETS.items():
		model = train_model(dataset_config)