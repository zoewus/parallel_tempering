import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from .model import MLP       
from .dataset import compute_mixture_pdf, compute_tsr_schedule  
from .config import DEVICE, DATASETS, CKPT_DIR, N_DIFFUSION_STEPS
from .sample import sampling

torch.manual_seed(42)
np.random.seed(42)

device = DEVICE
ckpt_dir = CKPT_DIR


# ---------- plotting + experiment runner (plot_samples, run_experiments) ----------
def load_model(path):
	"""Load trained model from checkpoint"""
	model = MLP().to(device)
	model.load_state_dict(torch.load(path, map_location=device))
	model.eval()
	return model


def samples_tuning(step_scale_list, n_langevin_steps_list, dataset_names, methods, k=1.0, sigma=1.0, x_limit=6):
	"""Generate samples and plot against true distribution on given axis"""
	samples = {}
	x_axis = np.linspace(-x_limit, x_limit, 500)
	bins = np.linspace(-x_limit, x_limit, 100)

	for name in dataset_names:
		model = load_model(f"{ckpt_dir}/{name}_1.0.pt")
		dataset_config = DATASETS[name]
		pdf = compute_mixture_pdf(dataset_config, x_axis, k)

		for method in methods:
			n_rows, n_cols = len(n_langevin_steps_list), len(step_scale_list)   # since 16 configs
			n_plots = n_rows * n_cols

			fig, axes = plt.subplots(n_rows, n_cols,
									 figsize=(4*n_cols, 3*n_rows),
									 sharex=True, sharey=True)
			
			overall_title = f"Tuning Comparison (k={k:.2f})"
			fig.suptitle(overall_title, fontsize=14, fontweight='bold')

			axes = axes.flatten()

			for i, step_scale in enumerate(step_scale_list):
				for j, n_langevin_steps in enumerate(n_langevin_steps_list):
					ax = axes[i * n_cols + j]

					x_sampled = sampling(
						model=model,
						dataset_config=dataset_config,
						method=method,
						k=k,
						sigma=sigma,
						step_scale=step_scale,
						n_langevin_steps=n_langevin_steps
					)
										
					ax.hist(
						x_sampled.cpu().numpy(),
						bins=bins,
						density=True,
						alpha=0.5,
						label=f"Samples (T = {k:.2f})"
					)
					ax.plot(x_axis, pdf, label="True PDF")
					
					base_title = f"{method} Sampling"
					if method.upper() in ["ULA", "MALA"]:
						langevin_info = f"\nsteps={n_langevin_steps:d}, step={step_scale:g}"
						title = base_title + langevin_info
					else:
						title = base_title

					ax.set_title(title, fontsize=8)
					ax.set_xlim(-x_limit, x_limit)
					print(f"method={method}, steps={n_langevin_steps:d}, step_scale={step_scale:g}")
					print("mean =", x_sampled.mean().item())
					print("var  =", x_sampled.var().item())
					samples[(step_scale, n_langevin_steps)] = x_sampled

			# Hide any unused axes if n_plots < n_rows*n_cols (not needed here but safe)
			for ax in axes[n_plots:]:
				ax.axis("off")

			fig.suptitle(f"{name} – {method} – T={k}", fontsize=14)
			fig.tight_layout(rect=[0, 0, 1, 0.97])

			os.makedirs("figures", exist_ok=True)
			out_path = os.path.join("figures", f"{name}_{method}_T{k:.2f}_tuning.png")
			fig.savefig(out_path, dpi=200)
			plt.close(fig)
			
	return x_sampled


def encode_dataset_name(name: str) -> str:
	# base mappings
	base_map = {
		"single": "s",
		"barrier": "b",
		"composed": "c",
		"single_overlapping": "so",
		"barrier_overlapping": "bo",
		"composed_overlapping": "co",
	}

	# default: use base map or fall back to the full name
	return base_map.get(name, name)


def plot_samples_grid(dataset_names, methods, k=1.0, sigma=1.0,  x_limit=6, save_dir="figures", figsize_per_panel=(5,4)):
	"""Generate samples and plot against true distribution"""
	os.makedirs(save_dir, exist_ok=True)

	n_rows = len(dataset_names)
	n_cols = len(methods)

	fig_width = figsize_per_panel[0] * n_cols
	fig_height = figsize_per_panel[0] * n_rows
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

	overall_title = f"Sampling Methods Comparison (k={k:.2f})"
	fig.suptitle(overall_title, fontsize=14, fontweight='bold')

	x_axis = np.linspace(-x_limit, x_limit, 500)
	bins = np.linspace(-x_limit, x_limit, 200)

	for i, name in enumerate(dataset_names):
		model = load_model(f"{ckpt_dir}/{name}_1.0.pt")
		dataset_config = DATASETS[name]

		pdf = compute_mixture_pdf(dataset_config, x_axis, k)

		step_scale = dataset_config["step_scale"]
		n_langevin_steps = dataset_config["n_langevin_steps"]

		samples = {}

		for j, method in enumerate(methods):

			ax = axes[i,j]

			x_sampled = sampling(
				model=model,
				dataset_config=dataset_config,
				method=method,
				k=k,
				sigma=sigma,
				step_scale=step_scale,
				n_langevin_steps=n_langevin_steps
			)

			samples[(name, method)] = x_sampled

			ax.hist(
				x_sampled.cpu().numpy(),
				bins=bins,
				density=True,
				alpha=0.5,
				label=f"Samples (T={k})"
			)

			ax.plot(x_axis, pdf, label="True PDF")

			base_title = f"{method}"
			if method.upper() in ["ULA", "MALA"]:
				# show Langevin specifics
				langevin_info = f" (steps = {n_langevin_steps:d}, step size = {step_scale:g})"
				title = base_title + langevin_info
			else:
				title = base_title

			ax.set_title(title)
			ax.set_xlim(-x_limit, x_limit)

	y_max = 0.0
	for row in axes:
		for ax in row:
			y_max = max(y_max, ax.get_ylim()[1])

	for row in axes:
		for ax in row:
			ax.set_ylim(0,y_max)
	
	plt.tight_layout()

	dataset_name_str = "_".join(encode_dataset_name(n) for n in dataset_names)
	method_str = "_".join(methods)
	filename = f"{dataset_name_str}_T{k:.2f}_{method_str}.png"
	save_path = os.path.join(save_dir, filename)

	plt.savefig(save_path, dpi=200)
	plt.show()

	return samples


# ---------- main() + CLI argument parsing ----------

if __name__ == "__main__":

	methods = ["MALA"]

	dataset_names = ["barrier", "composed"]

	step_scale_list = [0.001, 0.01, 0.1, 0.5]

	n_langevin_steps_list = [3, 5, 10, 15]

	for k in [4.0, 1.0, 0.25]:

		_ = samples_tuning(step_scale_list, n_langevin_steps_list, dataset_names, methods, k=k, sigma=None, x_limit=6)

		sys.stdout.flush()

		# _ = plot_samples_grid(dataset_names, methods, k=k, sigma=None, x_limit=12, save_dir="figures")