import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from config import DATASET_PATH, BATCH_SIZE

class CustomImageCaptionDataset(Dataset):
    def __init__(self, image_folder, caption_file, transform=None):
        self.image_folder = image_folder
        self.caption_file = caption_file
        self.transform = transform
        self.data = self._load_captions()

    def _load_captions(self):
        data = []
        with open(self.caption_file, 'r') as file:
            for line in file:
                image_name, caption = line.strip().split('\t')
                data.append((image_name, caption))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, caption = self.data[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, caption

def get_data_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomImageCaptionDataset(
        image_folder=DATASET_PATH / "images",
        caption_file=DATASET_PATH / "captions.txt",
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    return dataloader