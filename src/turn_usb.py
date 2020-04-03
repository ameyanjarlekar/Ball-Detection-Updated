#! /usr/bin/env python2
import pdb
import roslib
import rospy
import cv2
import numpy as np
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Image
import keras
from keras.models import load_model
from cv_bridge import CvBridge, CvBridgeError
import detect_ball
import time
import sys
import signal

#SIGINT handler
def sigint_handler(signal, frame):
    # print("Avg norm Error for new scheme: "+str(norm_err/count))
    # print("Avg norm Error for naive TD scheme"+str(norm_err_td/count))
    # print("Norm Improvements "+str(norm_impr/count))
    IMU_sub.unregister()
    pdb.set_trace()
    sys.exit(0)


def IMU_callback(inp):
	global curr_heading, enable_currheading
	if(not enable_currheading):
		enable_currheading=True
	curr_heading=np.argmin(np.abs((inp.data[2]*180/np.pi)-IMU_indice))

def supervised(array):
	# a = np.load("/home/ameya/Desktop/MRT/no_ball_h3.npy")
	# Input: np array of images Output: Max probab and index
	b = (new_model.predict(array, batch_size=None, verbose=0, steps=None))
	return np.argmax(b[:,0]),np.max(b[:,0])



if __name__ == '__main__':

	signal.signal(signal.SIGINT, sigint_handler)
	enable_gui = True 

	max_size = 255*1000 # max size of ball allowed
	#Number of images for one rotation (TODO: get this via IMU)
	one_rot = 180

	#Number of Pixels highlighted (output of unsupervised)
	mask_size = np.zeros(one_rot)
	#Ball Probability (output of supervised)
	ball_prob = np.zeros(one_rot)
	#List of np arrays: storing images
	data = np.zeros([one_rot,2])
	probab=np.zeros(one_rot)
	IMU_indice=np.linspace(-178,180,180)
	curr_heading=0
	enable_currheading=True
	#make ./file.h5
	#Feed Forward wts
	new_model = load_model('/home/rover-nuc/autonom_ws/src/Ball_Detection/src/new_model_num_coluoured.h5')
	rospy.init_node('turn usb ball detec node', anonymous=True)
	IMU_sub=rospy.Subscriber("IMU", Float32MultiArray, IMU_callback)

	resource = 0
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
		if rval and enable_currheading:
			cv2.imwrite("/home/ameya/Desktop/MRT/Current" + str(curr_heading) + ".jpg", frame)
			start = time.time()
			image, mask = detect_ball.detect_ball(frame) # get hysterisis mask
			
			if not np.any(mask):
				continue
			#Stores windows
			img_array = detect_ball.getMultiWindow(image, mask) # crop image and resize to 50x50
			#Stores best
			data[curr_heading] = supervised(img_array)
			probab[curr_heading]=data[curr_heading][1]


			print "prob", data[curr_heading], "\ttime", time.time() - start
			#cv2.imshow("mask", mask)
			#cv2.imshow("frame", image)
			#cv2.imshow("window",img_array[int(data[curr_heading][0])])
			# t = t + 1

			# if t == one_rot:
			# 	# np.save(str(no) + ".npy", data)
			# 	break
    		key = cv2.waitKey(20)
    		print("Ball most probably at:" + str(IMU_indice[np.argmax(probab)]) + "With Probability: "+ str(np.max(probab)))
	cv2.destroyWindow("preview")

# how to get output of supervised with sync
# how to store relevant data from output of unsup and sup
# how to determine whether other node has failed
# how to get direction data with sync
