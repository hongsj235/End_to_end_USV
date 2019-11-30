#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:43:19 2019

@author: seungjo
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim

import E2Emodel
import E2Edataset

from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# Data reading
dataroot = "/home/seungjo/catkin_ws/src/e2e_usv/Dataset/"
ckptroot = "/home/seungjo/catkin_ws/src/e2e_usv/checkpoints/"

# hyper-parameters setting
lr = 1e-3
weight_decay = 1e-5
batch_size = 16
num_workers = 4
test_size = 0.95
shuffle = True

epochs = 200
start_epoch = 0
resume = False

# Load dataset
trainset, valset = E2Edataset.load_data(dataroot, test_size)

# Get a data loader
print("==> Preparing dataset ...")
trainloader, validationloader = E2Edataset.data_loader(dataroot, trainset, valset, batch_size, shuffle, num_workers)

# 3: Define optimizer, model
model = E2Emodel.model()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()

# learning rate scheduler
scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

# transfer to gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device is: ', device)

# Preparing for saving train/valid loss to plot loss graph
f = open('/home/seungjo/catkin_ws/src/e2e_usv/loss.txt','w')

# 5: Train and validate network and save checkpoints
class Trainer(object):
    def __init__(self, ckptroot, model, device, epochs, criterion, optimizer, scheduler, start_epoch,
                 trainloader, validationloader, batch_size):

        super(Trainer, self).__init__()

        self.model = model
        self.device = device
        self.epochs = epochs
        self.ckptroot = ckptroot
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_epoch = start_epoch
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.batch_size = batch_size

    def train(self):
        """Training process."""
        self.model.to(self.device)
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            self.scheduler.step()
            
            # Training
            train_loss = 0.0
            self.model.train()

            for local_batch, (imgs, lefts, rights) in enumerate(self.trainloader):
                # Transfer to GPU
                imgs, lefts, rights = E2Edataset.toDevice(imgs, lefts, rights, self.device)
                # Model computations
                self.optimizer.zero_grad()
                total = [imgs, lefts, rights]
                datas = [total]
                # for data in datas:
                for i in datas:
                    imgs, lefts, rights = i
                    output_left = torch.zeros(imgs.shape[0]).to(device)
                    output_right = torch.zeros(imgs.shape[0]).to(device)
                    outputs = self.model(imgs)
                    for a in range(lefts.shape[0]):
                        temp_left = outputs[a][0]
                        temp_right = outputs[a][1]
                        output_left[a] = temp_left
                        output_right[a] = temp_right
                    loss1 = self.criterion(output_left, lefts)
                    loss2 = self.criterion(output_right, rights)
                    loss = loss1 + loss2
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.data.item()
                # t_loss.append(train_loss)
                if local_batch % 100 == 0:
                    print("Training Epoch: {} | Loss: {} | local_batch: {} ".format(epoch, train_loss / (local_batch + 1), local_batch))
                    f.write("Training Epoch: {} | Loss: {}".format(epoch, train_loss / (local_batch + 1)) +'\n')

            # Validation
            self.model.eval()
            valid_loss = 0
            with torch.set_grad_enabled(False):
                for local_batch, (imgs, lefts, rights) in enumerate(self.validationloader):
                    # Transfer to GPU
                    imgs, lefts, rights = E2Edataset.toDevice(imgs, lefts, rights, self.device)
                    # Model computations
                    self.optimizer.zero_grad()
                    total = [imgs, lefts, rights]
                    datas = [total]
                    for j in datas:
                        imgs, lefts, rights = j
                        output_left = torch.zeros(imgs.shape[0]).to(device)
                        output_right = torch.zeros(imgs.shape[0]).to(device)
                        outputs = self.model(imgs)
                        for a in range(outputs.shape[0]):
                            temp_left = outputs[a][0]
                            temp_right = outputs[a][1]
                            output_left[a] = temp_left
                            output_right[a] = temp_right
                    loss1 = self.criterion(output_left, lefts)
                    loss2 = self.criterion(output_right, rights)
                    loss = loss1 + loss2
                    valid_loss += loss.data.item()
                    # v_loss.append(valid_loss)
                    if local_batch % 100 == 0:
                        print("Validation Loss: {}".format(valid_loss / (local_batch + 1)))
                        f.write("Validation Loss: {}".format(valid_loss / (local_batch + 1)) +'\n')
            print()
            # Save model
            if epoch % 10 == 0 or epoch == self.epochs + self.start_epoch - 1:

                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }

                self.save_checkpoint(state)
        f.close()

    def save_checkpoint(self, state):
        """Save checkpoint."""
        print("==> Save checkpoint ...")
        if not os.path.exists(self.ckptroot):
            os.makedirs(self.ckptroot)

        torch.save(state, self.ckptroot + 'End_to_end_USV_model_{}.h5'.format(state['epoch']))



print("==> Start training ...")

# main function of training
trainer = Trainer(ckptroot, model, device, epochs, criterion, optimizer, scheduler, start_epoch, trainloader, validationloader, batch_size)
trainer.train()