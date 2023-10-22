import torch
from matplotlib import pyplot as plt

from diffusion_constants import steps, betas, sqrt_one_minus_alphas_cum_prod, sqrt_recip_alphas, posterior_variance
from diffusion_model.noise_forward_process import get_index_from_list, show_tensor_image


@torch.no_grad()
def sample_timestep(x, t, model):

    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cum_prod_t = get_index_from_list(
        sqrt_one_minus_alphas_cum_prod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cum_prod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(img_size, device, model):
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(steps / num_images)

    for i in range(0, steps)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model)
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            show_tensor_image(img.detach().cpu())

    plt.show()