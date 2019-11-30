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
            kernel size: 5x5, filter: 24, strides: 1x1, batch normalization, activation: ELU, maxpool(kernel size:2X2, stride:2) = 319 X 179 X 24
            kernel size: 5x5, filter: 36, strides: 1x1, activation: ELU, maxpool(kernel size:2X2, stride:2) = 158 X 88 X 36
            Drop out (0.2) 
            kernel size: 5x5, filter: 48, strides: 1x1, batch normalization, activation: ELU, maxpool(kernel size:2X2, stride:2) = 78 X 43 X 48
            kernel size: 3x3, filter: 64, strides: 1x1, activation: ELU, maxpool(kernel size:2X2, stride:2) = 39 X 21 X 64
            kernel size: 3x3, filter: 128, strides: 1x1, batch normalization, activation: ELU, maxpool(kernel size:2X2, stride:2) = 19 X 10 X 128
            Drop out (0.2) 
            kernel size: 3x3, filter: 128, strides: 1x1, activation: ELU, maxpool(kernel size:2X2, stride:2) = 9 X 5 X 128
            Drop out (0.1) 
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
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, padding=1),
            nn.BatchNorm2d(num_features=24),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 319*179*24
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 158*88*36
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, padding=1),
            nn.BatchNorm2d(num_features=48),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 78*43*48
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 39*21*64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 19*10*128
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 9*5*128
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=5760, out_features=1024),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=1024, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=16),
            nn.ELU(),
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



        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, padding=1)
        # self.batchnorm1 = nn.BatchNorm2d(num_features=24)
        # self.relu = nn.ReLU()
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, padding=1)
        # self.batchnorm2 = nn.BatchNorm2d(num_features=48)
        # self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.batchnorm3 = nn.BatchNorm2d(num_features=128)
        # self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.fc1 = nn.Linear(in_features=5760, out_features=1024)
        # self.fc2 = nn.Linear(in_features=1024, out_features=128)
        # self.fc3 = nn.Linear(in_features=128, out_features=12)
        # self.fc4 = nn.Linear(in_features=12, out_features=2)
        # self.dropout = nn.Dropout(p=0.1)
        # self.dropout_cnn = nn.Dropout2d(p=0.2)





        # x = x.view(x.size(0), 3, 360, 640)
        # x = self.pool(self.relu(self.batchnorm1(self.conv1(x))))
        # x = self.pool(self.relu(self.conv2(x)))
        # x = self.pool(self.relu(self.batchnorm2(self.conv3(x))))
        # x = self.dropout_cnn(x)
        # x = self.pool(self.relu(self.conv4(x)))
        # x = self.pool(self.relu(self.batchnorm3(self.conv5(x))))
        # x = self.dropout_cnn(x)
        # x = self.pool(self.relu(self.conv6(x)))
        # x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        # x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        # x = self.dropout(x)
        # outputs = self.fc4(x)
        # return outputs
