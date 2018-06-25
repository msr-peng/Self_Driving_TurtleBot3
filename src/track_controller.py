#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from self_driving_turtlebot3.msg import Lane

max_vel = 0.1
Kp = 10
Kd = 20
last_error = 0

def pid_controller(error):
	global last_error

	angular_z = Kp * error + Kd * (error - last_error)
	last_error = error

	linear_x = max_vel * ((1 - abs(error) / 0.1) ** 2)
	angular_z = -angular_z

	return linear_x, angular_z

def publish_vel(lane, args):

	vel_pub = args
	error = lane.deviation
	linear_x, angular_z = pid_controller(error)

	control = Twist()
	control.linear.x = linear_x
	control.linear.y = 0
	control.linear.z = 0
	control.angular.x = 0
	control.angular.y = 0
	control.angular.z = angular_z

	vel_pub.publish(control)
	rospy.loginfo("velocity update")

def get_error():

	rospy.init_node('get_error')

	lane_topic = "/lane_condition"

	vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
	rospy.Subscriber(lane_topic, Lane, publish_vel, callback_args=vel_pub)

	rospy.spin()

def get_shutdown():
	rospy.loginfo("Shutting down. cmd_vel will be zero")

	twist = Twist()
	twist.linear.x = 0
	twist.linear.y = 0
	twist.linear.z = 0
	twist.angular.x = 0
	twist.angular.y = 0
	twist.angular.z = 0

	vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
	vel_pub.publish(twist)

if __name__ == '__main__':
	vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
	try:
		get_error()
	except:
		print e

	finally:
		twist = Twist()
		vel_pub = rospy.Publisher("cmd_vel", Twist)
		twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
		twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
