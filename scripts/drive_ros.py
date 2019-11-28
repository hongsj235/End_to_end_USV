#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:43:54 2019

@author: seungjo
"""

#parsing command line arguments
#!/usr/bin/env python3
import os
import sys
import time
import rospy
import numpy as np
import cv2
import torch
import E2Edataset
import E2Emodel
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from sensor_msgs.msg import CompressedImage, Image, Float32
from matplotlib.ticker import NullLocator

image_array = None
flag = 0

class arg_parser(object):
    def __init__(self):
        self.weigts_path = None
        self.batch_size = None

def image_callback(self, data) :
    np_arr = np.frombuffer(data.data, np.uint8)
    image_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_array = cv2.resize(image_array, (360, 640))
    return image_array

def main(args):
	global image_arrayy
	global flag
	thrusters = thrust()
	rospy.init_node('e2e_drive', anonymous = True)
	opt = arg_parser()
	opt.weigts_path = rospy.get_param("~weigts_path")
	opt.batch_size = rospy.get_param("~batch_size")
    
    # Load pretrained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = E2Emodel.model().to(device)
    model.load_state_dick(torch.load(opt.weigts_path))

    # Load real-time images from the USV camera
    image_read = rospy.Subscriber('/wamv/sensors/cameras/middle_camera/image_raw/compressed', CompressedImage, image_callback, queue_size = 1)

    # Define publisher to USV thrusters
    left_pub = rospy.Publisher("left_cmd",Float32,queue_size=10)
    right_pub = rospy.Publisher("right_cmd",Float32,queue_size=10)

    left_thruster = Float32()
    right_thruster = Float32()

    while not rospy.is_shutdown():
        with torch.no_grad():
			# Input images to model
	        result = model(image_array)
	        left_thruster = result[0]
	        right_thruster = result[1]

	    	# Publish Thrusters value
            left_pub.publish(left_thruster)
            right_pub.publish(right_thruster)

            print([left_thruster, right_thruster])

	try:
		rospy.rostime.wallsleep(0.01)
	except KeyboardInterrupt:
		print("KeyboardInterrupt, Shutdown")

if __name__ =='__main__':
	main(sys.argv)