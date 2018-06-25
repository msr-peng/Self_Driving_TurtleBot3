#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image

bridge = CvBridge()
Matrix = np.array([[370.00135953,   0.        , 308.34269038],
				   [  0.        , 367.5902948 , 239.94645245],
				   [  0.        ,   0.        ,   1.        ]])
distCoeffs = np.array([[-0.46130559,  0.21223062, -0.00379475,  0.004662  , -0.04227536]])

def publish_undistort(image, args):
	undistort_pub = args
	cv2_img = bridge.imgmsg_to_cv2(image, "rgb8")
	rospy.loginfo("image received")
	undistorted = cv2.undistort(cv2_img, Matrix, distCoeffs, None, Matrix)
	undistorted = bridge.cv2_to_imgmsg(undistorted, "rgb8")

	undistort_pub.publish(undistorted)

	return

def get_distort():
	rospy.init_node('get_distort')

	image_topic = "/gamma"

	bridge = CvBridge()

	undistort_pub = rospy.Publisher("undistort", Image, queue_size=1)
	rospy.Subscriber(image_topic, Image, publish_undistort, callback_args=undistort_pub, queue_size=1)

	rospy.spin()

if __name__ == '__main__':
	get_distort()
