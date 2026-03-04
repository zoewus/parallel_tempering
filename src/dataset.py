import torch 
import numpy as np
from scipy.stats import norm
from .schedule import betas, alphas, alpha_bars, ts_desc

@torch.no_grad()
def generate_gaussian_mixture(n_samples, means, stds, device='cpu'):
    """Generates mixture of gaussians according to inputted means and standard deviations"""
    means = torch.as_tensor(means, dtype=torch.float32)
    n_gaussians = len(means)
    
    if isinstance(stds, (int, float)):
        stds = torch.full((n_gaussians,), float(stds))
    else:
        stds = torch.as_tensor(stds, dtype=torch.float32)
        assert len(stds) == n_gaussians, f"stds length {len(stds)} != n_gaussians {n_gaussians}"
        
    component_ids = np.random.choice(n_gaussians, size=n_samples)
    samples = torch.zeros(n_samples, 1, device=device)
    
    for i in range(n_gaussians):
        mask = component_ids == i
        samples[mask] = torch.normal(
			mean=float(means[i]),
			std=float(stds[i]),
			size=(mask.sum(), 1)
		).to(device)
    
    return samples


@torch.no_grad()
def compute_sigma_0(dataset_config):
    """Computes overall standard deviation for dataset according to dataset config parameters"""
    means = np.array(dataset_config['means'])
    stds = dataset_config['stds']
    
    if isinstance(stds, (int, float)):
        stds = np.full_like(means, stds, dtype=float)
    else:
        stds = np.array(stds)
    
    mean_of_mixture = np.mean(means)
    expected_x_squared = np.mean(stds**2 + means**2)
    variance = expected_x_squared - mean_of_mixture**2
    sigma_0 = np.sqrt(variance)
    
    return sigma_0


@torch.no_grad()
def compute_mixture_pdf(dataset_config, x_axis, temperature=1.0):
    """Computes analtytical pdf of training dataset from dataset config file, used for plotting"""
    means = np.array(dataset_config['means'])
    stds = dataset_config['stds']
    
    if isinstance(stds, (int, float)):
        stds = np.full(len(means), stds)
    else:
        stds = np.array(stds)
    
    temperature_scale = 1.0 / np.sqrt(temperature)
    stds = stds * temperature_scale
    
    n_gaussians = len(means)
    pdf = np.zeros_like(x_axis)
    
    for mu, sigma in zip(means, stds):
        pdf += norm.pdf(x_axis, loc=mu, scale=sigma)
    
    pdf /= n_gaussians  
    
    return pdf

@torch.no_grad()
def compute_tsr(t_batch, temperature, dataset_config):
	"""Computes temporal score rescaling coefficient"""
	sigma_0 = compute_sigma_0(dataset_config)
	sigma = sigma_0

	a_bar = alpha_bars[t_batch[0,0].long().item()]
	sigma_t = torch.sqrt(1.0 - a_bar)
	alpha_t = torch.sqrt(a_bar)

	eta_t = (alpha_t**2) / (sigma_t**2)
	num = eta_t * (sigma ** 2) + 1
	den = (eta_t * (sigma ** 2)) / temperature + 1
	#return num / den
	return 1