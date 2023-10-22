import os

from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class BirdsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_file)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
    ])

    data_dir = '../data/snakes_dataset'
    dataset = BirdsDataset(data_dir, transform=transform)

    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        print(batch.shape)

    batch = next(iter(dataloader))

    for image in batch:
        image = image.permute(1, 2, 0)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
