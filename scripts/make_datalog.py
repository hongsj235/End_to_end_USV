import os
import random
import csv

imglist = os.listdir('/home/seungjo/catkin_ws/src/e2e_usv/Dataset/image/')
imglist.sort()

datalog = open('/home/seungjo/catkin_ws/src/e2e_usv/Dataset/data.csv','w')

for i in range(len(imglist)):

	with open('/home/seungjo/catkin_ws/src/e2e_usv/Dataset/course1_wamv_thrusters_left_r_thrust_cmd.csv','r') as left_thrust:
		left_temp = left_thrust.readlines()[i]
		left =str(left_temp).split(',')
		left = left[1][0:-1]
		print(i)
		# for i in left_thrust.readlines():
		# 	left_temp = i.split(",")
		# 	left_temp = left_temp[1]
		# 	left_temp = left_temp[0:-1]
			# print(left_temp)

	with open('/home/seungjo/catkin_ws/src/e2e_usv/Dataset/course1_wamv_thrusters_right_r_thrust_cmd.csv','r') as right_thrust:
		right_temp = right_thrust.readlines()[i]
		right = str(right_temp).split(',')
		right = right[1][0:-1]
		print(i)
		# for i in right_thrust.readlines():
		# 	right_temp = i.split(",")
		# 	right_temp = right_temp[1]
		# 	right_temp = right_temp[0:-1]

	datalog.write("/home/seungjo/catkin_ws/src/e2e_usv/Dataset/image/"+imglist[i]+", "+left+", "+right+'\n')

datalog.close()
