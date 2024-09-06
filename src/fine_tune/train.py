import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import LEARNING_RATE, EPOCHS, CHECKPOINT_PATH
from dataset import get_data_loader
from model import ImageCaptioningModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageCaptioningModel('resnet50', 'bert-base-uncased', 128).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    dataloader = get_data_loader()

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for images, captions in tqdm(dataloader):
            images = images.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()
            outputs = model(images, captions)
            loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(dataloader)}")
        torch.save(model.state_dict(), CHECKPOINT_PATH / f"model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()