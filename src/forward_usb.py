#! /usr/bin/env python2

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
import keras
from keras.models import load_model
from cv_bridge import CvBridge, CvBridgeError
import detect_ball
import time
import os

max_size = 255*100000 # max size of ball allowed
full_size = 255*100000 # size of ball at minimum distance
#Number of images for one rotation (TODO: get this via IMU)
one_rot = 250
#Number of Pixels highlighted (output of unsupervised)
mask_size = np.zeros(one_rot)
#Ball Probability (output of supervised)
ball_prob = np.zeros(one_rot)
#List of np arrays: storing images
data = np.zeros([one_rot,2])

t = 0
new_model = load_model('/home/youknowwho/Documents/ROS/src/image_rcv/src/new_model_num_coluoured.h5')

def supervised(array):
	global new_model
	# a = np.load("/home/ameya/Desktop/MRT/no_ball_h3.npy")
	b = (new_model.predict(array, batch_size=None, verbose=0, steps=None))
	return np.argmax(b[:,0]),np.max(b[:,0])

# class publisher:

# 	def __init__(self):
# 		self.image_pub = rospy.Publisher("turn",Image,queue_size=10) # publish window to supervised
# 		self.bridge = CvBridge()
# 		self.image_sub = rospy.Subscriber("supervised",String,self.callback) # get probability from supervised

# 	def callback(self,data):
# 		global outstanding
# 		ball_prob.append(float(data.data))
# 		outstanding = 0


if __name__ == '__main__':
	# pub = publisher()
	# rospy.init_node('pub', anonymous=True)
	# r = rospy.Rate(1) # 1 second rate
	
	resource = 1
	print "Trying to open resource: " + str(resource)
	cap = cv2.VideoCapture(resource)
	# cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
	if not cap.isOpened():
		print "Error opening resource: " + str(resource)
		print "Maybe opencv VideoCapture can't open it"
		exit(0)

	print "Correctly opened resource, starting to show feed."
	while not rospy.is_shutdown():
		rval, frame = cap.read()
		if rval:
			# cv2.imwrite("/home/youknowwho/Documents/ROS/src/image_send/images/" + str(t) + ".jpg", frame)
			start = time.time()
			image, mask = detect_ball.detect_ball(frame) # get hysterisis mask

			if not np.any(mask):
				continue

			if np.sum(mask) > full_size:
				print "ball reached\t",np.sum(mask)
				break
			
			img_array = detect_ball.getMultiWindow(image, mask) # crop image and resize to 50x50
			data[t] = supervised(img_array)
			print "prob", data[t], "\ttime", time.time() - start
			cv2.imshow("mask", mask)
			cv2.imshow("frame", image)
			cv2.imshow("window",img_array[int(data[t][0])])
			# cv2.imwrite("/home/youknowwho/Documents/ROS/src/image_send/labled/" + str(t) + ".jpg", labled)
			# path = '/home/youknowwho/Documents/ROS/src/image_send/ball/' + str(t)
			# if not os.path.exists(path):
				# os.makedirs(path)
			# for j in range(len(data)):
				# cv2.imwrite(path + '/' + str(j) + ".jpg",data[j])

			# cv2.imshow("mask", mask)
			# cv2.imshow("frame", frame)

			# np.save(str(no) + ".npy", data)
			# s = time.time()
			# cv2.imshow("window", data[test(data)])
			# print "prob\t", test(data), '\t', testvalue(data)#, '\t', time.time()-s
    		key = cv2.waitKey(20)
	cv2.destroyWindow("preview")

# how to get output of supervised with sync
# how to store relevant data from output of unsup and sup
# how to determine whether other node has failed
# how to get direction data with sync

# stop on reaching ball