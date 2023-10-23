import torch

from diffusion_model.diffusion_constants import config
from diffusion_model.sampling import sample_plot_image
from diffusion_model.unet_backward_process import SimpleUnet

if __name__ == "__main__":
    training_parameters = config.get("training_parameters", {})
    img_size = training_parameters['img_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleUnet()
    checkpoint_path = "../models/model_epoch_10.pt"
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    sample_plot_image(img_size, device, model)