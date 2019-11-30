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

with open("/home/seungjo/catkin_ws/src/e2e_usv/loss.txt") as f:
    for line in f.readlines():
        if "Training" in line:
            training_loss.append(float(line.rstrip().split(": ")[-1]))
        elif "Validation" in line:
            validation_loss.append(float(line.rstrip().split(": ")[-1]))

training_loss = [float(loss) for loss in training_loss]
validation_loss = [float(loss) for loss in validation_loss]

training_loss = np.array(training_loss)
validation_loss = np.array(validation_loss)
print(len(training_loss))
print(len(validation_loss))


# Plot loss graph
plt.title("Training Loss vs Validation Loss")
plt.xlabel("Iterations")
plt.plot(range(len(training_loss)), training_loss, 'b', label='Training Loss')
plt.plot(np.linspace(0, len(training_loss), len(validation_loss)), validation_loss, 'g-.', label='Validation Loss')
plt.legend()
plt.grid(False)
plt.show()
