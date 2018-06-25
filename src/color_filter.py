#!/usr/bin/env python
import rospy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

color_offset = 25

def publish_filter(image, args):
	filter_pub = args

	yellow = bridge.imgmsg_to_cv2(image, "bgr8")

	rospy.loginfo("yellowed image received")

	B = yellow[:,:,0]
	G = yellow[:,:,1]
	R = yellow[:,:,2]

	B = B.astype(int)
	idx = B > 255
	B[idx] = 255
	B = np.uint8(B)

	G = G.astype(int)-1.5*color_offset
	idx = G < 0
	G[idx] = 0
	G = np.uint8(G)

	R = R.astype(int)-color_offset
	idx = R < 0
	R[idx] = 0
	R = np.uint8(R)

	result = np.zeros_like(yellow)
	result[:,:,0] = B
	result[:,:,1] = G
	result[:,:,2] = R

	result = bridge.cv2_to_imgmsg(result, "bgr8")

	filter_pub.publish(result)

	return

def get_yellow():
	rospy.init_node('get_yellow')

	image_topic = "/undistort"

	bridge = CvBridge()
	filter_pub = rospy.Publisher("color_filter", Image, queue_size=1)
	rospy.Subscriber(image_topic, Image, publish_filter, callback_args=filter_pub)

	rospy.spin()

if __name__ == '__main__':
	get_yellow()
