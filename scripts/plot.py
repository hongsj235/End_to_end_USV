#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:43:19 2019

@author: seungjo
"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns; sns.set(color_codes=True)

sns.set_style("whitegrid")

training_loss = []
validation_loss = []

with open("/home/seungjo/catkin_ws/src/e2e_usv/obstacle/loss.txt") as f:
	a = f.readlines()
	for i in range(50):
		if i == 0:
			train = a[1]
			training_loss.append(float(train.rstrip().split(": ")[-1]))
		else:
			train = a[2+(22*i)]
			training_loss.append(float(train.rstrip().split(": ")[-1]))

with open("/home/seungjo/catkin_ws/src/e2e_usv/obstacle/loss.txt") as f:
	for i in range(50):
		if i == 0:
			valid = a[18]
			validation_loss.append(float(valid.rstrip().split(": ")[-1]))
		else:
			valid = a[19+(22*i)]
			validation_loss.append(float(valid.rstrip().split(": ")[-1]))

training_loss = [float(loss) for loss in training_loss]
validation_loss = [float(loss) for loss in validation_loss]

training_loss = np.array(training_loss)
validation_loss = np.array(validation_loss)

# Plot loss graph
plt.title("Training Loss vs Validation Loss")
plt.xlabel("Epoches")
plt.plot(range(len(training_loss)), training_loss, 'b', label='Training Loss')
plt.plot(np.linspace(0, len(training_loss), len(validation_loss)), validation_loss, 'g-.', label='Validation Loss')
plt.legend()
plt.grid(False)
plt.show()
