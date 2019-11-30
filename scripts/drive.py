#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:43:54 2019

@author: seungjo
"""

#parsing command line arguments
#!/usr/bin/env python3
import sys
# import rospy
import torch
import E2Edataset
import E2Emodel

# Parameters setting
image_array = None
flag = 0

weigts_path = "/home/seungjo/catkin_ws/src/e2e_usv/checkpoints/model_obstacle.h5"
image_root = '/home/seungjo/catkin_ws/src/e2e_usv/Dataset/'
batch_size = 1
num_workers = 4
test_size = 0.8
shuffle = True
trainset, valset = E2Edataset.load_data(image_root, test_size)
trainloader, validationloader = E2Edataset.data_loader(image_root, trainset, valset, batch_size, shuffle, num_workers)

# Main function
def main(args):
	global validationloader
	global batch_size
	global valset
	global weigts_path
	global image_array
	global flag
    
    # Load pretrained weights
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = E2Emodel.model().to(device)
	ckpt = torch.load(weigts_path)
	model.load_state_dict(ckpt['state_dict'])

	left_thruster = []
	right_thruster = []
	thruster = []

	for local_batch, (imgs, lefts, rights) in enumerate(validationloader):
		imgs, lefts, rights = E2Edataset.toDevice(imgs, lefts, rights, device)

		with torch.no_grad():
			result = model(imgs)
			# print(result)
			left_thruster = result[0][0]
			right_thruster = result[0][1]
			thruster.append([left_thruster, right_thruster])

	f = open('/home/seungjo/catkin_ws/src/e2e_usv/result.txt', 'w')
	for i in range(len(thruster)):
		left = float(thruster[i][0])
		right = float(thruster[i][1])
		f.write(str(valset[i][0]) +', '+str(left)+', '+str(right)+'\n')
	f.close()

if __name__ =='__main__':
	main(sys.argv)