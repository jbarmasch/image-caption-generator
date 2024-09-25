import os
import csv
import zipfile
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from more_itertools import ilen

class CustomDataset():
    def __init__(self, name = "nlphuji/flickr30k", split='test', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), transform=transforms.ToTensor(), split_ratios=(0.8, 0.1, 0.1)):
        self.device = device

        # Load dataset with just the "test" split
        self.dataset = load_dataset(name, split=split, streaming=True)

        # Create empty lists for train, validation, and test
        train_filter = lambda x: x['split'] == 'train'
        val_filter = lambda x: x['split'] == 'val'
        test_filter = lambda x: x['split'] == 'test'

        print(type(self.dataset))
        self.train_dataset = self.dataset.filter(train_filter)
        self.val_dataset = self.dataset.filter(val_filter)
        self.test_dataset = self.dataset.filter(test_filter)
        print("Dataset filetered")

        self.train_len = None
        self.val_len = None
        self.test_len = None
    
    def get_train_len(self):
        if self.train_len is None:
            self.train_len = ilen(self.train_dataset)
        return self.train_len
    
    def get_val_len(self):
        if self.val_len is None:
            self.val_len = ilen(self.val_dataset)
        return self.val_len
    
    def get_test_len(self):
        if self.test_len is None:
            self.test_len = ilen(self.test_dataset)
        return self.test_len
    
    def get_datasets(self):
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_dataloaders(self, batch_size, collate_fn):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=collate_fn)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, collate_fn=collate_fn)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, collate_fn=collate_fn)
        return train_loader, val_loader, test_loader