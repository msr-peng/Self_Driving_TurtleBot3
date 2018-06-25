#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import UInt8
from geometry_msgs.msg import Twist
from self_driving_turtlebot3.msg import Lane
from self_driving_turtlebot3.msg import WhichLane

max_vel = 0.05
Kp = 20
Kd = 40
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
trigger_40 = 0

flag_14 = False
flag_18 = False
flag_35 = False
flag_38 = False
flag_39 = False
flag_40 = False
past_flag_38 = False
past_flag_39 = False

interim = False
interim_start = 0

def pid_controller(error):
	global last_error

	angular_z = Kp * error + Kd * (error - last_error)
	last_error = error

	linear_x = max_vel * ((1 - abs(error) / 0.1) ** 2)
	angular_z = -angular_z

	return linear_x, angular_z

def publish_vel(lane, args):
	global trigger_14
	global trigger_18
	global trigger_40

	global flag_14
	global flag_18
	global flag_40

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

	time_now = rospy.get_time()
	# Stop
	if flag_14:
		control.linear.x = 0
		control.angular.z = 0

		if (time_now-trigger_14) > 3:
			flag_14 = False
	# General Caution
	if flag_18:
		control.linear.x = 0.33 * control.linear.x

		if (time_now-trigger_18) > 5:
			flag_18 = False
	# Roundabout Mandatory
	if flag_40:
		control.linear.x = 0
		control.angular.z = 1

		if (time_now-trigger_40) > 3:
			flag_40 = False

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
	global trigger_40

	global flag_14
	global flag_18
	global flag_35
	global flag_38
	global flag_39
	global flag_40

	global past_flag_38
	global past_flag_39

	global interim
	global interim_start

	whichlane_pub = args
	signal = type.data

	before_flag_38 = flag_38
	before_flag_39 = flag_39
	# Stop
	if signal == 14:
		current_14 = rospy.get_time()
		if (current_14 - past_14) > 2:
			sign_14 = 1
		elif (current_14 - trigger_14) > 10:
			sign_14 += 1
		past_14 = current_14
	# General Caution
	if signal == 18:
		current_18 = rospy.get_time()
		if (current_18 - past_18) > 2:
			sign_18 = 1
		elif (current_18 - trigger_18) > 10:
			sign_18 += 1
		past_18 = current_18
	# Ahead Only
	if signal == 35:
		current_35 = rospy.get_time()
		if (current_35 - past_35) > 2:
			sign_35 = 1
		else:
			sign_35 += 1
		past_35 = current_35
	# Keep Right
	if signal == 38:
		current_38 = rospy.get_time()
		if (current_38 - past_38) > 2:
			sign_38 = 1
		else:
			sign_38 += 1
		past_38 = current_38
	# Keep Left
	if signal == 39:
		current_39 = rospy.get_time()
		if (current_39 - past_39) > 2:
			sign_39 = 1
		else:
			sign_39 += 1
		past_39 = current_39
	# Roundabout Mandatory
	if signal == 40:
		current_40 = rospy.get_time()
		if (current_40 - past_40) > 2:
			sign_40 = 1
		elif (current_40 - trigger_40) > 10:
			sign_40 += 1
		past_40 = current_40

	if sign_14 > 3:
		rospy.loginfo("do Stop")
		trigger_14 = rospy.get_time()
		sign_14 = 0
		flag_14 = True
	if sign_18 > 2:
		rospy.loginfo("do Deceleration")
		trigger_18 = rospy.get_time()
		sign_18 = 0
		flag_18 = True
	if sign_35 > 3:
		rospy.loginfo("do Follow center of track")
		sign_35 = 0
		flag_35 = True
		flag_38 = False
		flag_39 = False
	if sign_38 > 3:
		if flag_38 == False:
			past_flag_38 = flag_38
			past_flag_39 = flag_39
			rospy.loginfo("do Follow the right lane")
			sign_38 = 0
			sign_39 = 0
			sign_40 = 0
			flag_35 = False
			flag_38 = True
			flag_39 = False
	if sign_39 > 3:
		if flag_39 == False:
			past_flag_38 = flag_38
			past_flag_39 = flag_39
			rospy.loginfo("do Follow the left lane")
			sign_38 = 0
			sign_39 = 0
			sign_40 = 0
			flag_35 = False
			flag_38 = False
			flag_39 = True
	if sign_40 > 4:
		rospy.loginfo("do Turn behind")
		sign_40 = 0
		trigger_40 = rospy.get_time()
		flag_40 = True

	after_flag_38 = flag_38
	after_flag_39 = flag_39

	if (before_flag_38 != after_flag_38) or (before_flag_39 != after_flag_39):
		past_flag_38 = before_flag_38
		past_flag_39 = before_flag_39

	whichlane = WhichLane()

	if not interim:
		# if the lane keeping skip the status of center:
		if (past_flag_38 != flag_38) and (past_flag_39 != flag_39):
			interim = True
			interim_start = rospy.get_time()
		else:
			whichlane.right = flag_38
			whichlane.left  = flag_39

	if interim:
		interim_current = rospy.get_time()
		if (interim_current - interim_start < 2):
			whichlane.right = False
			whichlane.left  = False
		else:
			interim = False
			past_flag_38 = False
			past_flag_39 = False

	whichlane_pub.publish(whichlane)

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
	whichlane_pub = rospy.Publisher("which_lane", WhichLane)
	rospy.Subscriber(lane_topic, Lane, publish_vel, callback_args=vel_pub)
	rospy.Subscriber(sign_topic, UInt8, judge_sign, callback_args=whichlane_pub)

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