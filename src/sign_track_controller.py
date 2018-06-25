#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import UInt8
from geometry_msgs.msg import Twist
from self_driving_turtlebot3.msg import Lane

max_vel = 0.1
Kp = 10
Kd = 20
last_error = 0

sign_14 = 0
sign_18 = 0
sign_35 = 0
sign_38 = 0
sign_39 = 0
sign_40 = 0

past_14 = 0
past_18 = 0
past_35 = 0
past_38 = 0
past_39 = 0
past_40 = 0

current_14 = 0
current_18 = 0
current_35 = 0
current_38 = 0
current_39 = 0
current_40 = 0

trigger_14 = 0
trigger_18 = 0

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
	# rospy.loginfo("velocity update")

def judge_sign(type, args):
	global sign_14
	global sign_18
	global sign_35
	global sign_38
	global sign_39
	global sign_40

	global past_14
	global past_18
	global past_35
	global past_38
	global past_39
	global past_40

	global trigger_14
	global trigger_18

	behavior_pub = args
	signal = type.data

	if signal == 14:
		current_14 = rospy.get_time()
		if (current_14 - past_14) > 3:
			sign_14 = 1
		elif (current_14 - trigger_14) > 10:
			sign_14 += 1
		past_14 = current_14

	if signal == 18:
		current_18 = rospy.get_time()
		if (current_18 - past_18) > 3:
			sign_18 = 1
		elif (current_18 - trigger_18) > 10:
			sign_18 += 1
		past_18 = current_18

	if signal == 35:
		current_35 = rospy.get_time()
		if (current_35 - past_35) > 3:
			sign_35 = 1
		else:
			sign_35 += 1
		past_35 = current_35

	if signal == 38:
		current_38 = rospy.get_time()
		if (current_38 - past_38) > 3:
			sign_38 = 1
		else:
			sign_38 += 1
		past_38 = current_38

	if signal == 39:
		current_39 = rospy.get_time()
		if (current_39 - past_39) > 3:
			sign_39 = 1
		else:
			sign_39 += 1
		past_39 = current_39

	if signal == 40:
		current_40 = rospy.get_time()
		if (current_40 - past_40) > 3:
			sign_40 = 1
		else:
			sign_40 += 1
		past_40 = current_40

	if sign_14 > 3:
		rospy.loginfo("do Stop")
		trigger_14 = rospy.get_time()
		sign_14 = 0
	if sign_18 > 3:
		rospy.loginfo("do Deceleration")
		trigger_18 = rospy.get_time()
		sign_18 = 0
	if sign_35 > 3:
		rospy.loginfo("do Follow center of track")
		trigger_35 = rospy.get_time()
		sign_35 = 0
	if sign_38 > 3:
		rospy.loginfo("do Follow the right lane")
	if sign_39 > 3:
		rospy.loginfo("do Follow the left lane")
	if sign_40 > 3:
		rospy.loginfo("do Turn behind")

	behavior_pub.publish(sign_35)

def get_error():

	global past_14, current_14
	global past_18, current_18
	global past_35, current_35
	global past_38, current_38
	global past_39, current_39
	global past_40, current_40

	rospy.init_node('get_error')

	past_14 = rospy.get_time()
	past_18 = rospy.get_time()
	past_35 = rospy.get_time()
	past_38 = rospy.get_time()
	past_39 = rospy.get_time()
	past_40 = rospy.get_time()

	current_14 = rospy.get_time()
	current_18 = rospy.get_time()
	current_35 = rospy.get_time()
	current_38 = rospy.get_time()
	current_39 = rospy.get_time()
	current_40 = rospy.get_time()

	lane_topic = "/lane_condition"
	sign_topic = "/sign_types"

	vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
	behavior_pub = rospy.Publisher("behavior", UInt8)
	rospy.Subscriber(lane_topic, Lane, publish_vel, callback_args=vel_pub)
	rospy.Subscriber(sign_topic, UInt8, judge_sign, callback_args=behavior_pub)

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
	try:
		get_error()
	except rospy.ROSInterruptException:
		get_shutdown()

# 14: Stop
# 18: General caution
# 35: Ahead only
# 38: Keep right
# 39: Keep left
# 40: Roundabout mandatory