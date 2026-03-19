import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from .model import MLP       
from .dataset import compute_mixture_pdf, compute_tsr_schedule  
from .schedule import betas, alphas, alpha_bars, ts_desc
from .config import DEVICE, DATASETS, CKPT_DIR, N_DIFFUSION_STEPS

torch.manual_seed(42)
np.random.seed(42)

device = DEVICE
ckpt_dir = CKPT_DIR

# ---------- sampling utilities (compute_score, compute_tsr, etc.) ----------
@torch.no_grad()
def compute_score(model, x, t, k, sigma, dataset_config):
	"""Computes score = - epsilon / √(1 - α_bar)"""
	temp_schedule = compute_tsr_schedule(k, sigma, dataset_config)
	temp_t = temp_schedule[t]

	t_batch = t * torch.ones_like(x)
	eps_hat = model(x, t_batch)
	a_bar = alpha_bars[t]

	return -eps_hat * temp_t / torch.sqrt(1.0 - a_bar)


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
def compute_log_transition_ratio(model, x, x_hat, t, step_size, k, sigma, dataset_config):
	"""Computes log [ k(x | x_hat) / k(x_hat | x) ]"""
	score_x = compute_score(model, x, t, k, sigma, dataset_config) 
	score_x_hat = compute_score(model, x_hat, t, k, sigma, dataset_config) 

	forward_diff = x_hat - x - step_size * score_x 
	forward_sq = forward_diff**2
	
	backward_diff = x - x_hat - step_size * score_x_hat 
	backward_sq = backward_diff**2
	
	return (forward_sq - backward_sq) / (4.0 * step_size)


@torch.no_grad()
def compute_score_integral(model, x, x_hat, t, k, sigma, dataset_config, n_segments=10):
	"""Computes energy of a noise-based model thrfugh integration"""
	s = torch.linspace(0.0, 1.0, n_segments, device=device)
	
	r = r_curve_func(x, x_hat, s)
	r_deriv = r_deriv_func(x, x_hat, s)
	
	r_flat = r.reshape(-1, 1)
	score_hat = compute_score(model, r_flat, t, k, sigma, dataset_config).reshape(r.shape[0], -1) 

	integrand = - score_hat * r_deriv
	f = -torch.trapz(integrand, s, dim=1).unsqueeze(-1)
	
	return f


@torch.no_grad()
def compute_correction(model, x, x_hat, t, step_size, k, sigma, dataset_config):
	"""Computes acceptance rate for MALA and returns corrected x"""
	f = compute_score_integral(model, x, x_hat, t, k, sigma, dataset_config)
	log_transition_ratio = compute_log_transition_ratio(model, x, x_hat, t, step_size, k, sigma, dataset_config)
	sigma_t = torch.sqrt(1 - alpha_bars[t])

	a = torch.clamp(torch.exp(f + log_transition_ratio), max=1.0) # note: we do f and not f/sigma_t because sigma term already in score_hat
	u = torch.rand_like(a)
	accept_mask = (u < a).float()
	x = accept_mask * x_hat + (1 - accept_mask) * x

	return x, accept_mask


@torch.no_grad()
def metric_tensor_patch(tau, score_patch):
	"""Computes metric tensor for a single chunk of the samples"""
	bs_increment, s = score_patch.shape
	outer_product = torch.einsum("bc,bd-> cd", score_patch, score_patch) / (bs_increment * s)

	G =  torch.eye(s, device = device) + tau * outer_product

	eigvals, eigvecs = torch.linalg.eigh(G)
	eigvals_inv = eigvals.mean().item() / (eigvals)

	G_inv = eigvecs @ torch.diag(eigvals_inv) @ eigvecs.T
	new_score_patch = torch.einsum('ij,bj->bi', G_inv, score_patch) 

	return new_score_patch


def metric_tensor(tau, score):
	"""Computes scaled score for all samples"""
	bs, s = score.shape # score is of shape (bs, 1)
	num_patches = 8
	increment = bs // num_patches # divide particles into the number of batches we want

	for i in range(num_patches): # goes through each patch
	
		h_start = i * increment

		score_patch = score[:, :, h_start : h_start + increment ].detach().clone()

		new_score_patch = metric_tensor_patch(tau, score_patch)

		score[:, :, h_start : h_start + increment ] = new_score_patch

	return score


# ---------- main sampling function (sampling) ----------
@torch.no_grad()
def sampling(model, dataset_config, method, k=1.0, sigma=1.0, step_scale=None, n_langevin_steps=None):
	"""Sampling algorithm for DDPM, ULA, and MALA"""
	model.eval()

	n_samples = dataset_config["n_samples"]
	x = torch.randn(n_samples, 1, device=device)

	for t in ts_desc:
		alpha_t = alphas[t]
		beta_t = betas[t]
		sqrt_alpha_t = torch.sqrt(alpha_t)
		sqrt_beta_t = torch.sqrt(beta_t)

		if method in ["DDPM"]:
			score_hat = compute_score(model, x, t, k, sigma, dataset_config)
			noise = torch.randn_like(x)
			x = (x + beta_t * score_hat) / sqrt_alpha_t + sqrt_beta_t * noise

		elif method in ["ULA", "MALA"]:
			score_hat = compute_score(model, x, t, k, sigma, dataset_config)
			noise = torch.randn_like(x)
			x = (x + beta_t * score_hat) / sqrt_alpha_t + sqrt_beta_t * noise

			step_size = beta_t * torch.tensor(step_scale)

			for langevin_step in range(n_langevin_steps):

				score_hat = compute_score(model, x, t, k, sigma, dataset_config)
				noise = torch.randn_like(x)
				x_hat = x + step_size * score_hat + torch.sqrt(2.0 * step_size) * noise	

				if method == "ULA":
					x = x_hat
				
				elif method == "MALA":
					x, a = compute_correction(model, x, x_hat, t, step_size, k, sigma, dataset_config)
					if t in [90, 50, 10, 1] and langevin_step == 0:
						print(f"t={t}, beta={betas[t]:.6f}, step_size={step_size:.6f}, steps={n_langevin_steps:d}")
						print(f"  accept magnitude: {a.abs().mean():.4f}")
						print(f"  score magnitude: {score_hat.abs().mean():.4f}")
						print(f"  noise magnitude: {noise.abs().mean():.4f}")
						print(f"  update magnitude: {(step_size * score_hat).abs().mean():.4f}")

		else:
			raise ValueError(f"Unknown method: {method}. Expected 'DDPM', 'ULA', or 'MALA'.")

	return x