from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")


class IncorrectDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return (image, label)


def data_loader(images, labels, batch_size, shuffle=True):
    dataset = []
    for i in range(len(images)):
        dataset.append(tuple([images[i], labels[i]]))

    data_loader = torch.utils.data.DataLoader(dataset=IncorrectDataset(dataset),
                                              batch_size=batch_size, shuffle=shuffle)

    return data_loader
