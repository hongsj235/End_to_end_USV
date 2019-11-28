#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:48:06 2019

@author: seungjo
"""
import os 
import torch
import torch.nn as nn


class model(nn.Module):

    def __init__(self):
        """Initialize model.
            Image normalization to avoid saturation and make gradients work better.
            kernel size: 3x3, filter: 32, strides: 1x1, batch normalization, activation: ELU, maxpool(kernel size:2X2, stride:2)
            kernel size: 3x3, filter: 48, strides: 1x1, activation: ELU
            kernel size: 3x3, filter: 64, strides: 1x1, batch normalization, activation: ELU, maxpool(kernel size:2X2, stride:2)
            kernel size: 3x3, filter: 128, strides: 1x1, batch normalization, activation: ELU
            kernel size: 3x3, filter: 64, strides: 1x1, activation: ELU, maxpool(kernel size:2X2, stride:2)
            Drop out (0.2)
            Fully connected: neurons: 1024, activation: ELU
            Fully connected: neurons: 128, activation: ELU
            Fully connected: neurons: 16, activation: ELU
            Fully connected: neurons: 2 (output)
        
        the convolution layers are meant to handle feature engineering.
        the fully connected layer for predicting the steering angle.
        the elu activation function is for taking care of vanishing gradient problem.
        """
        super(model, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(num_features=32),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 48, 3),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.BatchNorm2d(num_features=64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(num_features=128),
            nn.ELU(),
            nn.Conv2d(128, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=199424, out_features=1024),
            nn.ELU(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=16),
            nn.Linear(in_features=16, out_features=2)
        )

    def forward(self, input):
        """Forward pass."""
        input = input.view(input.size(0), 3, 360, 640)
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output