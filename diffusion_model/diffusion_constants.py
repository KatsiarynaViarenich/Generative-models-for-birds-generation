import numpy as np
import torch
import yaml

with open('config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

diffusion_betas_parameters = config.get("diffusion_betas_parameters", {})

steps = diffusion_betas_parameters.get("steps", 200)
beta_start = diffusion_betas_parameters.get("start", 0.004)
beta_end = diffusion_betas_parameters.get("end", 0.06)
betas = torch.linspace(beta_start, beta_end, steps)
alphas = 1. - betas
cumulative_alphas = torch.cumprod(alphas, dim=0)
# alphas_cum_prod_prev = F.pad(alphas_cum_prod[:-1], (1, 0), value=1.)
prev_cumulative_alphas = np.concatenate(([1.], cumulative_alphas[:-1]))
sqrt_recip_alphas = torch.sqrt(1. / alphas)
sqrt_alphas_cum_prod = torch.sqrt(cumulative_alphas)
sqrt_one_minus_alphas_cum_prod = torch.sqrt(1. - cumulative_alphas)
posterior_variance = betas * (1. - prev_cumulative_alphas) / (1. - cumulative_alphas)