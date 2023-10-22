import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_model.diffusion_constants import steps
from diffusion_model.noise_forward_process import forward_diffusion_sample
from diffusion_model.sampling import sample_plot_image
from diffusion_model.birds_dataset import BirdsDataset
from diffusion_model.transforms import get_transforms
from diffusion_model.unet_backward_process import SimpleUnet


def get_loss(model, x_0, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


def train():
    with open('config.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    training_parameters = config.get("training_parameters", {})

    epochs = training_parameters['epochs']
    img_size = training_parameters['img_size']
    batch_size = training_parameters['batch_size']

    data_dir = '../data/birds_dataset/images'
    data_transforms = get_transforms(img_size)
    dataset = BirdsDataset(data_dir, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleUnet()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for step, batch in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()

            t = torch.randint(0, steps, (batch_size,), device=device).long()
            loss = get_loss(model, batch, t, device)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                sample_plot_image(img_size, device, model)


if __name__ == "__main__":
    train()