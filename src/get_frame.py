#!/usr/bin/env python
import rospy
import sys
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np
import math
from cv_bridge import CvBridge, CvBridgeError
import cv2

bridge = CvBridge()

def foo1 (i, L=[]):
	if len(L)==0:
		L.append(0)
	L[0]+=i
	return L[0]

def foo2 (i, L=[]):
	if len(L)==0:
		L.append(0)
	L[0]+=i
	return L[0]

def process_frame(image):
	foo2(1)
	if ((foo2(0)%20) == 0):
		print("Received the %dth image!" %foo1(1))
		try:
			# Convert your ROS Image message to OpenCV2
			cv2_img = bridge.imgmsg_to_cv2(image, "bgr8")
		except CvBridgeError, e:
			print(e)
		else:
			# Save your OpenCV2 image as a png
			cv2.imwrite('camera_image_%d.png' %foo1(0), cv2_img)

def get_frame():
	rospy.init_node('get_frame', anonymous=True)

	image_topic = "/raw_image"
	
	rospy.Subscriber(image_topic, Image, process_frame)

	rospy.spin()

if __name__ == '__main__':
	get_frame()
