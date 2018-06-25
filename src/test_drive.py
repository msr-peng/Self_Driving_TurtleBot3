#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from self_driving_turtlebot3.msg import Lane

if __name__=="__main__":
	rospy.init_node('test_drive')
	pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

	target_linear_vel = 0
	target_angular_vel = 0.5
	control_linear_vel = 0
	control_angular_vel = 0
	try:
		while not rospy.is_shutdown():
			if target_linear_vel > control_linear_vel:
				control_linear_vel = min(target_linear_vel, control_linear_vel + (0.01/4.0))
			else:
				control_linear_vel = target_linear_vel

			if target_angular_vel > control_angular_vel:
				control_angular_vel = min(target_angular_vel, control_angular_vel + (0.1/4.0))
			else:
				control_angular_vel = target_angular_vel

			twist = Twist()
			twist.linear.x = control_linear_vel; twist.linear.y = 0; twist.linear.z = 0
			twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = control_angular_vel
			pub.publish(twist)

	except:
		print e

	finally:
		twist = Twist()
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        pub.publish(twist)