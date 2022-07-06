import os
import sys
import json
import torch
import random
import argparse
import itertools
import torchvision
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .mlp import MLP
from .training import ClassifierTrainer
from .classifier import ArgMaxClassifier, BinaryClassifier, select_roc_thresh

class generate_data():
    def __init__(self, amount, upper_range, length):
        self.length = length
        self.amount = amount
        self.upper_range = upper_range
    def generate(self):
        X = torch.randint(self.upper_range, (self.amount, self.length), dtype = torch.long)
        y, _ = torch.sort(X)
        return (X, y)
    
#custom dataset for torch
class CustomDataset(Dataset):
    def __init__(self, data_class):
        self.sequences, self.labels = data_class.generate()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.sequences[idx]
        label = self.labels[idx]
        return x, label