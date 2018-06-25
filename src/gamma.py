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

gamma = 1
invGamma = 1.0/gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

def publish_gamma(image, args):
	gamma_pub = args

	dark = bridge.imgmsg_to_cv2(image, "bgr8")

	rospy.loginfo("dark image received")

	gamma = cv2.LUT(dark, table)

	result = bridge.cv2_to_imgmsg(gamma, "bgr8")

	gamma_pub.publish(result)

	return

def get_dark():
	rospy.init_node('get_dark')

	image_topic = "/raw_image"

	bridge = CvBridge()
	gamma_pub = rospy.Publisher("gamma", Image, queue_size=1)
	rospy.Subscriber(image_topic, Image, publish_gamma, callback_args=gamma_pub)

	rospy.spin()

if __name__ == '__main__':
	get_dark()
