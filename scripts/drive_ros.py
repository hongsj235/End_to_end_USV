#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:43:54 2019

@author: seungjo
"""

#parsing command line arguments
#!/usr/bin/env python3
import sys
import rospy
import numpy as np
import cv2
import torch
import E2Emodel

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

image_array = np.zeros([1,720,1280,3], dtype="uint8")
publish_array = np.zeros([1,360,640,3], dtype="uint8")
flag = 0

class arg_parser(object):
    def __init__(self):
        self.weights_path = None
        self.batch_size = None

def image_callback(data) :
    global image_array
    global flag
    flag = 0
    np_arr = np.frombuffer(data.data, np.uint8)
    image_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_array = cv2.resize(image_array, (360, 640))
    image_array = np.array([image_array])
    flag = 1

def main(args):
    global image_array
    global publish_array
    global flag
    rospy.init_node('e2e_drive', anonymous = True)
    # Load real-time images from the USV camera
    image_read = rospy.Subscriber('/wamv/sensors/cameras/middle_camera/image_raw/compressed', CompressedImage, image_callback, queue_size = 1)
    rate = rospy.Rate(10)

    opt = arg_parser()
    opt.weights_path = rospy.get_param("~weights_path")
    opt.batch_size = rospy.get_param("~batch_size")
    
    # Load pretrained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = E2Emodel.model().to(device)
    ckpt = torch.load(opt.weights_path)
    model.load_state_dict(ckpt['state_dict'])

    # Define publisher to USV thrusters
    left_pub = rospy.Publisher("/wamv/thrusters/left_r_thrust_cmd", Float32, queue_size=1)
    right_pub = rospy.Publisher("/wamv/thrusters/right_r_thrust_cmd", Float32, queue_size=1)

    left_thruster = Float32()
    right_thruster = Float32()

    while not rospy.is_shutdown():
        while flag == 0:
            pass
        publish_array = image_array.copy()
        publish_array = torch.FloatTensor(publish_array).to(device)
        with torch.no_grad():
            result = model(publish_array)
            left_thruster = float(result[0][0])
            right_thruster = float(result[0][1])
            
            print('left_thruster : '+str(left_thruster)+'\t'+ 'right_thruster : ' + str(right_thruster) + '\n')

            # Publish Thrusters value
            left_pub.publish(left_thruster)
            right_pub.publish(right_thruster)

    try:
        rospy.rostime.wallsleep(0.01)
    except KeyboardInterrupt:
        print("KeyboardInterrupt, Shutdown")

if __name__ =='__main__':
    main(sys.argv)