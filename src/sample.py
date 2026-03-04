import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from .model import MLP       
from .dataset import compute_mixture_pdf, compute_tsr  
from .schedule import betas, alphas, alpha_bars, ts_desc
from .config import DEVICE, DATASETS, CKPT_DIR

torch.manual_seed(42)
np.random.seed(42)

device = DEVICE
ckpt_dir = CKPT_DIR

# ---------- sampling utilities (compute_score, compute_tsr, etc.) ----------
@torch.no_grad()
def compute_score(model, x, t_batch):
	"""Computes score = - epsilon / √(1 - α_bar)"""
	eps_hat = model(x, t_batch)
	a_bar = alpha_bars[t_batch[0, 0].long().item()]
	return -eps_hat / torch.sqrt(1.0 - a_bar)


@torch.no_grad()
def r_curve_func(x, x_hat, s):
	"""Computes curve where r(0) = x and r(1) = x hat"""
	return x + s * (x_hat - x)


@torch.no_grad()
def r_deriv_func(x, x_hat, s):
	"""Computes analytical derivative of r curve"""
	ones = torch.ones_like(s)
	return ones * (x_hat - x)


@torch.no_grad()
def compute_log_transition_ratio(model, x, x_hat, t_batch, step_size):
	"""Computes log [ k(x | x_hat) / k(x_hat | x) ]"""
	score_x = compute_score(model, x, t_batch)
	score_x_hat = compute_score(model, x_hat, t_batch)

	forward_diff = x_hat - x - step_size * score_x
	forward_sq = forward_diff**2
	
	backward_diff = x - x_hat - step_size * score_x_hat
	backward_sq = backward_diff**2
	
	return (forward_sq - backward_sq) / (4.0 * step_size)


@torch.no_grad()
def compute_score_integral(model, x, x_hat, t_batch, n_segments):
	"""Computes energy of a noise-based model through integration"""
	s = torch.linspace(0.0, 1.0, n_segments, device=device)
	
	r = r_curve_func(x, x_hat, s)
	r_deriv = r_deriv_func(x, x_hat, s)
	
	r_flat = r.reshape(-1, 1)
	t_flat = t_batch[0].item() * torch.ones_like(r_flat)
	epsilon = model(r_flat, t_flat).reshape(r.shape[0], -1)
	
	integrand = epsilon * r_deriv
	f = -torch.trapz(integrand, s, dim=1).unsqueeze(-1)
	
	return f


@torch.no_grad()
def compute_correction(model, x, x_hat, t_batch, step_size, num_segments=10):
	"""Computes acceptance rate for MALA and returns corrected x"""
	f = compute_score_integral(model, x, x_hat, t_batch, num_segments)
	log_transition_ratio = compute_log_transition_ratio(model, x, x_hat, t_batch, step_size)
	sigma_t = torch.sqrt(1 - alpha_bars[t_batch[0].item()])

	a = torch.clamp(torch.exp(f / sigma_t + log_transition_ratio), max=1.0)
	u = torch.rand_like(a)
	accept_mask = (u < a).float()
	x = accept_mask * x_hat + (1 - accept_mask) * x

	# print(f"Acceptance:            {a.mean().item()}")

	return x, a

def energy_gradient_func(model, x, t_batch):
	t = t_batch[0, 0].long().item()
	eps_hat = model(x, t_batch)
	alpha_t = alphas[t]
	beta_t = betas[t]
	a_bar = alpha_bars[t]
	sqrt_alpha_t = torch.sqrt(alpha_t)
	sqrt_beta_t = torch.sqrt(beta_t)
	return  ((1/sqrt_beta_t)-(1/torch.sqrt( 1 - a_bar))) * eps_hat / sqrt_alpha_t

def energy_gradient_func_2(model, x, t_batch):
	"""Computes score = - epsilon / √(1 - α_bar)"""
	eps_hat = model(x, t_batch)
	a_bar = alpha_bars[t_batch[0, 0].long().item()]
	return -eps_hat / (1.0 - a_bar)

# ---------- main sampling function (sampling) ----------
@torch.no_grad()
def sampling(model, dataset_config, method, step_scale=None, n_langevin_steps=None, temperature=1.0, temperature_schedule=None):
	"""Sampling algorithm for DDPM, ULA, and MALA"""
	model.eval()

	n_samples = dataset_config["n_samples"]

	x = torch.randn(n_samples, 1, device=device)
	ones = torch.ones(n_samples, 1, device=device)

	for t in ts_desc:
		t_batch = (t * ones).long()
		if temperature_schedule is not None:
			temperature = temperature_schedule[t]
				  
		alpha_t = alphas[t]
		beta_t = betas[t]
		alpha_bar_t = alpha_bars[t]
		sqrt_alpha_t = torch.sqrt(alpha_t)
		sqrt_beta_t = torch.sqrt(beta_t)

		score_hat = compute_score(model, x, t_batch)
		noise = torch.randn_like(x)

		x = (x + beta_t * score_hat) / sqrt_alpha_t + sqrt_beta_t * noise

		if method in ["ULA", "MALA"]:

			step_size = beta_t * torch.tensor(step_scale)

			for _ in range(n_langevin_steps):

				noise = torch.randn_like(x)
				eps_hat = - model(x, t_batch) / torch.sqrt(1 - alpha_bar_t)
				x_hat = x + step_size * eps_hat + torch.sqrt(2.0 * step_size) * noise	
				
				if method == "MALA":
					x, a = compute_correction(model, x, x_hat, t_batch, step_size)

						# Add this inside the ULA loop
					if t in [90, 50, 10, 1] and _ == 0:
						print(f"t={t}, beta={betas[t]:.6f}, step_size={step_size:.6f}, steps={n_langevin_steps:.1f}")
						print(f"  accept magnitude: {a.abs().mean():.4f}")
						print(f"  score magnitude: {score_hat.abs().mean():.4f}")
						print(f"  noise magnitude: {noise.abs().mean():.4f}")
						print(f"  update magnitude: {(step_size * score_hat).abs().mean():.4f}")
				elif method == "ULA":
					x = x_hat
				
		else:
			raise ValueError(f"Unknown method: {method}. Expected 'DDPM', 'ULA', or 'MALA'.")

	return x

# ---------- plotting + experiment runner (plot_samples, run_experiments) ----------
def load_model(path):
	"""Load trained model from checkpoint"""
	model = MLP().to(device)
	model.load_state_dict(torch.load(path, map_location=device))
	model.eval()
	return model


def plot_samples_ax(ax, config, method,
					step_scale=None, n_langevin_steps=None, x_limit=6, temperature=1.0, temperature_schedule=None):
	"""Generate samples and plot against true distribution on given axis"""
	name, dataset_config = config

	model = load_model(f"{ckpt_dir}/{name}_{temperature}.pt")
	x_sampled = sampling(
		model=model,
		dataset_config=dataset_config,
		method=method,
		step_scale=step_scale,
		n_langevin_steps=n_langevin_steps,
		temperature=temperature,
		
	)
	
	x_axis = np.linspace(-x_limit, x_limit, 500)
	pdf = compute_mixture_pdf(dataset_config, x_axis, temperature)
	bins = np.linspace(-x_limit, x_limit, 100)
	
	ax.hist(
		x_sampled.cpu().numpy(),
		bins=bins,
		density=True,
		alpha=0.5,
		label=f"Samples (T = {temperature:.1f})"
	)
	ax.plot(x_axis, pdf, label="True PDF")
	
	base_title = f"{method} Sampling"
	if method.upper() in ["ULA", "MALA"]:
		langevin_info = f"\nsteps={n_langevin_steps}, step={step_scale:g}"
		title = base_title + langevin_info
	else:
		title = base_title

	ax.set_title(title, fontsize=8)
	ax.set_xlim(-x_limit, x_limit)
	ax.legend(fontsize=6)

	return x_sampled


def plot_samples(config, method, step_scale=None, n_langevin_steps=None, x_limit=6, save_dir="figures", temperature=1.0, temperature_schedule=None):
	"""Generate samples and plot against true distribution"""
	name, dataset_config = config

	os.makedirs(save_dir, exist_ok=True)

	model = load_model(f"{ckpt_dir}/{name}_{temperature}.pt")
	x_sampled = sampling(
		model=model,
		dataset_config=dataset_config,
		method=method,
		step_scale=step_scale,
		n_langevin_steps=n_langevin_steps,
		temperature=temperature,
		temperature_schedule=temperature_schedule
	)
	
	x_axis = np.linspace(-x_limit, x_limit, 500)
	pdf = compute_mixture_pdf(dataset_config, x_axis, temperature)
	
	bins = np.linspace(-x_limit, x_limit, 100)
	
	plt.figure(figsize=(5, 4))
	plt.hist(
		x_sampled.cpu().numpy(),
		bins=bins,
		density=True,
		alpha=0.5,
		label=f"Samples (T = {temperature:.1f})"
	)
	plt.plot(x_axis, pdf, label="True PDF")
	
	# Build title (and possibly subtitle info) depending on method
	base_title = f"{method} Sampling"
	if method.upper() in ["ULA", "MALA"]:
		# show Langevin specifics
		langevin_info = f" (steps = {n_langevin_steps}, step size = {step_scale:g})"
		title = base_title + langevin_info
	else:
		title = base_title

	plt.title(title)
	plt.xlim(-x_limit, x_limit)
	plt.legend()
	plt.tight_layout()

	step_str = f"_steps{n_langevin_steps}" if n_langevin_steps is not None else ""
	scale_str = f"_step{step_scale:g}" if step_scale is not None else ""

	filename = f"{name}_T{temperature:.1f}_{method}{step_str}{scale_str}.png"
	save_path = os.path.join(save_dir, filename)

	plt.savefig(save_path, dpi=200)
	plt.show()

	return x_sampled

# ---------- main() + CLI argument parsing ----------

if __name__ == "__main__":
	methods = ["ULA"]
	dataset_names = ["barrier"]
	temperature = 1.0
	temperature_schedule = None
	steps = [20, 25, 30, 50, 60]              # increases going down
	step_sizes = [0.05, 0.1, 0.2, 0.3, 0.4] # increases going right

	param_grid = [(s, h) for s in steps for h in step_sizes]

	for method in methods:
		for name in dataset_names:
			dataset_config = DATASETS[name]
			config = (name, dataset_config)

			n_plots = len(param_grid)
			n_rows, n_cols = len(steps), len(step_sizes)   # since 16 configs

			fig, axes = plt.subplots(n_rows, n_cols,
									 figsize=(4*n_cols, 3*n_rows),
									 sharex=True, sharey=True)
			axes = axes.flatten()

			for idx, ((n_steps, scale), ax) in enumerate(zip(param_grid, axes)):
				samples = plot_samples_ax(
					ax=ax,
					config=config,
					method=method,
					step_scale=scale,
					n_langevin_steps=n_steps,
					temperature=temperature,
					temperature_schedule=temperature_schedule,
				)
				print(f"method={method}, steps={n_steps}, step_scale={scale}")
				print("mean =", samples.mean().item())
				print("var  =", samples.var().item())

			# Hide any unused axes if n_plots < n_rows*n_cols (not needed here but safe)
			for ax in axes[n_plots:]:
				ax.axis("off")

			fig.suptitle(f"{name} – {method} – T={temperature}", fontsize=14)
			fig.tight_layout(rect=[0, 0, 1, 0.97])

			os.makedirs("figures", exist_ok=True)
			out_path = os.path.join("figures", f"{name}_{method}_T{temperature:.1f}_grid_pgrad_3_beta.png")
			fig.savefig(out_path, dpi=200)
			plt.close(fig)