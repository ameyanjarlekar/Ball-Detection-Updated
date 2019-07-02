#! /usr/bin/env python2
import Queue
import sys
import cv2
import numpy as np
import keras
from keras.models import load_model
#from cv_bridge import CvBridge, CvBridgeError
#import detect_ball
import time
import signal
import imutils

#SIGINT handler
def sigint_handler(signal, frame):
	#Do something while breaking
	pdb.set_trace()
	sys.exit(0)


class Ball_Detec:

	def __init__(self):
		self.resource = 0
		self.max_size = 255*100000 # max size of ball allowed
		self.full_size = 255*100000 # size of ball at minimum distance
		self.high_thresh = {'low':(28,100,120), 'high':(48,200,230)}
		self.low_thresh = {'low':(28,0,80), 'high':(57,255,255)}
		self.high_thresh_sunny = {'low':(20,60,120), 'high':(48,200,230)}
		self.low_thresh_sunny = {'low':(20,0,80), 'high':(57,255,255)}
		self.Visited = np.zeros((1,1))
		self.low_mask = np.zeros((1,1))
		self.clusterQ = Queue.Queue()
		# high and low colour boundaries in RGB 152 85.0 78.4

	def open_stream(self):
		print ("Trying to open resource: ") + str(self.resource)
		self.stream = cv2.VideoCapture(self.resource)
		# cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
		if not self.stream.isOpened():
			print ("Error opening resource: ") + str(self.resource)
			print ("Maybe opencv VideoCapture can't open it")

		print ("Correctly opened resource ID:" +str(self.resource)+", starting to show feed.")
		
	def read_stream(self,unsupervised_mode="hough",save=False,debugging=True):
		while True:
			rval, frame = self.stream.read()
			if rval:
				if(save):
					self.save_img("Filename")

				image, mask = self.get_mask(frame) # get hysterisis mask
				if not np.any(mask):
					continue

				if np.sum(mask) > self.full_size:
					print ("ball reached\t",np.sum(mask))
					break

				img_array = self.getMultiWindow(image, mask) # crop image and resize to 50x50
				
				if(debugging):
					cv2.imshow("mask", mask)
					cv2.imshow("frame", image)
					print(np.shape(img_array))

				
				i = 0
				while i < np.shape(img_array)[0]:
					gray = cv2.cvtColor(img_array[i], cv2.COLOR_BGR2GRAY)
					i = i+1 

				print(np.shape(gray))

				if(unsupervised_mode=="hough"):
					data[t] = self.hough(gray)
				# elif (unsupervised_mode=="GMM"):
				# 	data[t] = self.GMM(gray)

				key = cv2.waitKey(20)
				cv2.destroyWindow("preview")
			else:
				print("Stream Read RVal False: Unable to get frames")

	def hough(self,array):
		cv2.imwrite("/home/ameya/Desktop/MRT/curr" + '/' + "imp" + ".jpg",array)
		circles = cv2.HoughCircles(array,cv2.HOUGH_GRADIENT,1,20,param1=40,param2=25,minRadius=5,maxRadius=30)
		print(circles)
		if circles is None:
			a = 0
		else:		
			circles = np.uint16(np.around(circles))
			print(circles)
			for i in circles[0,:]:
				# draw the outer circle
				cv2.circle(array,(i[0],i[1]),i[2],(0,255,0),2)
				# draw the center of the circle
				cv2.circle(array,(i[0],i[1]),2,(0,0,255),3)

			cv2.imshow('detected circles',array)
			cv2.imwrite("/home/ameya/Desktop/MRT/curr" + '/' + str(i) + ".jpg",array)
			#cv2.waitKey(0)
			cv2.destroyAllWindows()
		print("printed")

	def getMultiWindow(self,image, mask):
		windows = []
		mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
		if not np.any(mask):
			window = self.getWindow(image, mask)
			windows.append(window)
			return np.array(windows)
		
		ret, labels = cv2.connectedComponents(mask)
		
		# xrange is just faster implementation of range
		for i in xrange(1,ret):
			label = cv2.inRange(labels,i,i)
			# print np.any(label)
			# cv2.imshow(str(i),label)
			# cv2.waitKey()
			# cv2.destroyWindow("preview")
			window = self.getWindow(image, label)
			windows.append(window)
		return np.array(windows)#,labeled_img

	def getWindow(self,image, mask):
		if (np.any(mask)):
			x,y,w,h = cv2.boundingRect(mask)
			# print x,y,w,h,image.shape
			s = max(w,h)
			if (s < 40): s = 40

			x1 = x-5
			y1 = y-5
			x2 = x+5+s
			y2 = y+5+s

			if x1 < 0:
				x2 = x2 - x1
				x1 = 0
			if y1 < 0:
				y2 = y2 - y1
				y1 = 0
			if x2 >= image.shape[1]:
				x1 = max(0,x1-(x2-image.shape[1]))
				x2 = image.shape[1]-1
			if y2 >= image.shape[0]:
				y1 = max(0,y1-(y2-image.shape[0]))
				y2 = image.shape[0]-1

			image = image[y1:y2,x1:x2]
			# print x1,x2,y1,y2
			# print
		window = cv2.resize(image,(50,50))
		# cv2.imshow("wind",window)
		# kernel = np.zeros( (9,9), np.float32)
		# kernel[4,4] = 2.0   #Identity, times two! 
		# boxFilter = np.ones( (9,9), np.float32) / 81.0
		# kernel = kernel - boxFilter
		# window = cv2.filter2D(window, -1, kernel)
		# cv2.rectangle(image,(x1,y1),(x2,y2),(255,255,255),2)
		return window



	def save_img(self,filename):
		cv2.imwrite(filename, self.frame)

	########################## debugging #################################
	# high_thresh = {'low':(28,100,120), 'high':(48,200,230)}
	# low_thresh = {'low':(28,0,80), 'high':(57,255,255)}
	######################################################################

	def get_mask(frame,gaussian=False):
		# shrink image
		shrink_img = imutils.resize(frame, width=600)

		# BGR to HSV
		hsv_image = cv2.cvtColor(shrink_img, cv2.COLOR_BGR2HSV)
		print("gauss",gaussian)
		if(gaussian):
			hsv_image = cv2.GaussianBlur(hsv_image, (11, 11), 0)

		masked_high = cv2.inRange(hsv_image, self.high_thresh['low'], self.high_thresh['high'])
		masked_low = cv2.inRange(hsv_image, self.low_thresh['low'], self.low_thresh['high'])
		if (not np.any(masked_high)):
			# print "calc sunny"
			masked_high = cv2.inRange(hsv_image, self.high_thresh_sunny['low'], self.high_thresh_sunny['high'])
			masked_low = cv2.inRange(hsv_image, self.low_thresh_sunny['low'], self.low_thresh_sunny['high'])

		hyst_mask = self.hyst_threshold(masked_high,masked_low)
		return shrink_img, hyst_mask

	def hyst_threshold(self,mask_high,mask_low):
		self.Visited = np.zeros(mask_low.shape)
		self.low_mask = mask_low
		for i in range(mask_high.shape[0]):
			for j in range(mask_high.shape[1]):
				if (self.Visited[i,j]==0 and mask_high[i,j]>0):
					self.clusterQ.put((i,j))
					while (not self.clusterQ.empty()):
						self.cluster()
		return self.Visited.astype(np.uint8)


	def cluster(self):
		(i,j) = self.clusterQ.get()
		if (i<0 or j<0 or i>=self.Visited.shape[0] or j>=self.Visited.shape[1]):
			return

		if (self.Visited[i,j]>0):
			return
		if (self.low_mask[i,j]==0):
			return

		self.Visited[i,j] = 255
		self.clusterQ.put((i-1,j))
		self.clusterQ.put((i,j-1))
		self.clusterQ.put((i,j+1))
		self.clusterQ.put((i+1,j))


if __name__ == '__main__':
	np.set_printoptions(threshold='nan')
	signal.signal(signal.SIGINT, sigint_handler)
	Ball_Detec_obj=Ball_Detec()
	Ball_Detec_obj.open_stream()
	Ball_Detec_obj.read_stream()


	

# how to get output of supervised with sync
# how to store relevant data from output of unsup and sup
# how to determine whether other node has failed
# how to get direction data with sync

# stop on reaching ball


# Can take probabilities of 5-10 images (setting frame rate high) and get the median probability and take the angle of the image at the centre of the sample of 10 images. This would help reduce noisy predictions. 