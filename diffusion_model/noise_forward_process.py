import torch

import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from diffusion_model.diffusion_constants import sqrt_alphas_cum_prod, sqrt_one_minus_alphas_cum_prod
from diffusion_model.birds_dataset import BirdsDataset
from diffusion_model.transforms import get_reverse_transforms


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cum_prod_t = get_index_from_list(sqrt_alphas_cum_prod, t, x_0.shape)
    sqrt_one_minus_alphas_cum_prod_t = get_index_from_list(sqrt_one_minus_alphas_cum_prod, t, x_0.shape)
    return sqrt_one_minus_alphas_cum_prod_t.to(device) * x_0.to(device) + sqrt_alphas_cum_prod_t.to(device) * noise.to(device), noise.to(device)


def show_tensor_image(image):
    reverse_transforms = get_reverse_transforms()
    # Take first image of batch
    # TODO: test with single image
    if len(image.shape) == 4:
        image = image[0, :, :, :]
        print("HERE was the image")
    plt.imshow(reverse_transforms(image))


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    data_dir = '../data/birds_dataset/images'
    dataset = BirdsDataset(data_dir, transform=transform)

    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    batch = next(iter(dataloader))
    batch = batch.to(device)
    x_0 = batch[0]
    x_0 = x_0.reshape(1, 3, 96, 96)
    x_t, noise = forward_diffusion_sample(x_0, torch.tensor([150]), device)

    show_tensor_image(x_t[0])