#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:48:36 2019

@author: seungjo
"""

import os 
import cv2
import torch

import numpy as np
import csv
import pandas as pd

from scipy import signal
from torch.utils import data
from torch.utils.data import DataLoader

def toDevice(imgs, left, right, device):
    """Enable cuda."""
    return imgs.float().to(device), left.float().to(device), right.float().to(device)

def augment(current_image, left, right):
    if np.random.rand() < 0.5:
        current_image = cv2.flip(current_image, 1)
        temp_left = left
        temp_right = right
        left = temp_right
        right = temp_left
    return current_image, left, right

def load_data(data_dir, test_size):
    """Load training data and train validation split"""
    # reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(data_dir, 'data.csv'), names=['image', 'left_thruster', 'right_thruster'])

    # smooth data signal with `savgol_filter`
    data_df["left_thruster"] = signal.savgol_filter(data_df["left_thruster"].values.tolist(), 51, 11)
    data_df["right_thruster"] = signal.savgol_filter(data_df["right_thruster"].values.tolist(), 51, 11)

    # Divide the data into training set and validation set
    train_len = int(test_size * data_df.shape[0])
    valid_len = data_df.shape[0] - train_len
    trainset, valset = data.random_split(data_df.values.tolist(), lengths=[train_len, valid_len])
    return trainset, valset

class AugmentDataset(data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        batch_samples = self.samples[index]
        imgName, left, right = batch_samples
        img = cv2.imread(imgName)
        img = cv2.resize(img, dsize=(360, 640), interpolation=cv2.INTER_AREA)
        if img is None:
            print(name)
        # pdb.set_trace()
        if batch_samples[1] != batch_samples[2] : # if left/right thruster value is not same
            img, left, right = augment(img, left, right)
        # img = self.transform(img)
        img = img/127.5 - 1.0
        return (img, left, right)
      
    def __len__(self):
        return len(self.samples)

class Dataset_val(data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        batch_samples = self.samples[index]
        imgName, left, right = batch_samples
        img = cv2.imread(imgName)
        img = cv2.resize(img, dsize=(360, 640), interpolation=cv2.INTER_AREA)
        if img is None:
            print(name)
        # img = self.transform(img)
        img = img/127.5 - 1.0
        return (img, left, right)
      
    def __len__(self):
        return len(self.samples)

def data_loader(dataroot, trainset, valset, batch_size, shuffle, num_workers):
    """Self-Driving vehicles simulator dataset Loader.

    Args:
        trainset: training set
        valset: validation set
        batch_size: training set input batch size
        shuffle: whether shuffle during training process
        num_workers: number of workers in DataLoader

    Returns:
        trainloader (torch.utils.data.DataLoader): DataLoader for training set
        testloader (torch.utils.data.DataLoader): DataLoader for validation set
    """
    transformations = None

    # Load training data and validation data
    training_set = AugmentDataset(trainset, transformations)
    trainloader = DataLoader(training_set,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)

    validation_set = Dataset_val(valset, transformations)
    valloader = DataLoader(validation_set,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers)

    return trainloader, valloader