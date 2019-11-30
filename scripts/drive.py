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
# import rospy
import numpy as np
import cv2
import torch
import E2Edataset
import E2Emodel
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
# from sensor_msgs.msg import CompressedImage, Image, Float32

image_array = None
flag = 0


image_root = '/home/seungjo/catkin_ws/src/e2e_usv/Dataset/'
batch_size = 10
num_workers = 4
test_size = 0.8
shuffle = True
trainset, valset = E2Edataset.load_data(image_root, test_size)
trainloader, validationloader = E2Edataset.data_loader(image_root, trainset, valset, batch_size, shuffle, num_workers)


class arg_parser(object):
    def __init__(self):
        self.weigts_path = None
        self.batch_size = None
        self.image_folder = None

# def image_callback(self, data) :
#     np_arr = np.frombuffer(data.data, np.uint8)
#     image_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     image_array = cv2.resize(image_array, (360, 640))
#     return image_array

def main(args):
	global validationloader
	global batch_size
	global valset

	global image_array
	global flag
	# thrusters = thrust()
	# rospy.init_node('e2e_drive', anonymous = True)
	opt = arg_parser()
	opt.weigts_path = "/home/seungjo/catkin_ws/src/e2e_usv/checkpoints/End_to_end_USV_model_50.h5"
	opt.batch_size = batch_size
	opt.image_folder = image_root
    
    # Load pretrained weights
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = E2Emodel.model().to(device)
	ckpt = torch.load(opt.weigts_path)
	model.load_state_dict(ckpt['state_dict'])
	# model.state_dict(torch.load(opt.weigts_path))

    # Load real-time images from the USV camera
#    image_read = rospy.Subscriber('/wamv/sensors/cameras/middle_camera/image_raw/compressed', CompressedImage, image_callback, queue_size = 1)

    # Define publisher to USV thrusters
#    left_pub = rospy.Publisher("left_cmd",Float32,queue_size=10)
#    right_pub = rospy.Publisher("right_cmd",Float32,queue_size=10)

	left_thruster = []
	right_thruster = []
	thruster = []

	for local_batch, (imgs, lefts, rights) in enumerate(validationloader):
        # Transfer to GPU
		imgs, lefts, rights = E2Edataset.toDevice(imgs, lefts, rights, device)
		total = [imgs, lefts, rights]
		datas = [total]
        # for data in datas:
		with torch.no_grad():
			for i in datas:
				imgs, lefts, rights = i
				result = model(imgs)
				# print(result)
				left_thruster = result[0][0]
				right_thruster = result[0][1]
				thruster.append([left_thruster, right_thruster])

	# print(thruster)
	f = open('/home/seungjo/catkin_ws/src/e2e_usv/result.txt', 'w')
	for i in range(len(thruster)):
		left = float(thruster[i][0])
		right = float(thruster[i][1])
		f.write(str(valset[i][0]) +', '+str(left)+', '+str(right)+'\n')
	f.close()

 #    while not rospy.is_shutdown():
 #    	with torch.no_grad():
	# 		# Input images to model
	#     	result = model(image_array)
	#     	left_thruster = result[0]
	#     	right_thruster = result[1]

	#     	# Publish Thrusters value
	#     	left_pub.publish(left_thruster)
	#     	right_pub.publish(right_thruster)

	#     	print([left_thruster, right_thruster])

	# try:
	# 	rospy.rostime.wallsleep(0.01)
	# except KeyboardInterrupt:
	# 	print("KeyboardInterrupt, Shutdown")

if __name__ =='__main__':
	main(sys.argv)