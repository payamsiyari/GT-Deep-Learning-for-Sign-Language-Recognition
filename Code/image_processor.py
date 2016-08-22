import cv2
from cv2 import cv
from cv2.cv import *
import numpy as np 
import random
import os
import cPickle as pickle
import copy

def writePickle(struct, filename):
	file1 = open(filename,"wb") 			
	pickle.dump(struct,file1)
	file1.close()

def cropImage(img,xlower,xupper,ylower,yupper):
	crop_img = img[ylower:yupper, xlower:xupper] 
	return crop_img

def sp_noise(image,prob):
	'''
	Add salt and pepper noise to image
	prob: Probability of the noise
	'''
	output = np.zeros(image.shape,np.uint8)
	thres = 1 - prob 
	for i in range(image.shape[0]):
	    for j in range(image.shape[1]):
	        rdn = random.random()
	        if rdn < prob:
	            output[i][j] = 0
	        elif rdn > thres:
	            output[i][j] = 255
	        else:
	            output[i][j] = image[i][j]
	return output

def rotateImage(image, angle):
	image_center = tuple(np.array(image.shape)/2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle,1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
	return result


def showImage(img):
	cv2.imshow('rotated image',im1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

path = '/net/ht140/payam-hadar/DL/dataSnippet'
classes = []
height = 200 						#change this to match what you are going to call while generating input
width = 200
startFileNumber = 25 				#skipping the first 25 image files and the last 20 : see line 76
classDict = {}
numofclasses = 9
numofangles = 2  								#This is equivalent to 4 angles (5 degree either side)
numofnoise = 5 									#5 noise images - probability for noise is 0.05 (see code below)
skipImages = 4
for dirs in os.walk(path): 									#all the directories in that path
	dirnames = dirs[1:][0]
	classes = copy.copy(dirnames)
	for dirn in dirnames: 									#All the words in sign language
		if dirn.lower() not in classDict:
			print 'Doing word',dirn.lower()
			classDict[dirn.lower()] = {} 					#upper most level of class dict is the class; the word in sign language
			dirs = []
			for dirname2 in os.walk(path+'/'+dirn): 		#Iterating through the people 1
				dirs = dirname2[1]
				break
			for dirname3 in dirs: 								#list of all the people who signed that word
				person = dirname3.split('_')[0].lower() 		#The person's name
				classDict[dirn.lower()][person] = {}
				filenumber = 500
				for filenames in os.walk(path+'/'+dirn+'/'+dirname3): 		#lowest level ie the frames for each person
					filenumber = len(filenames[2])
				upper = {}
				lower = {}
				print 'Doing',filenumber,'images for person',person
				for num in xrange(startFileNumber,filenumber-20,skipImages): 						#iterating through all the images in order
					filename = str(num)+'.jpg'
					im = cv2.imread(path+'/'+dirn+'/'+dirname3+'/'+filename)
					gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 			#CONVERT TO GRAYSCALE IMAGE 
					
					if 'original' not in upper:
						upper['original'] = []

					if 'original' not in lower:
						lower['original'] = []

					upImage = cropImage(gray_image,0,gray_image.shape[1],20,gray_image.shape[0]/2.0)
					upImage = cv2.resize(upImage, (width,height))
					upper['original'].append(upImage) 			

					lowImage = cropImage(gray_image,0,gray_image.shape[1],gray_image.shape[0]/2.0+20,gray_image.shape[0])
					lowImage=  cv2.resize(lowImage, (width,height))
					lower['original'].append(lowImage) 			

					'''
					cv2.imshow("cropped", upper['original'][-1])
					cv2.waitKey(0)

					cv2.imshow("cropped",lower['original'][-1])
					cv2.waitKey(0)
					'''

					
					if 'rotational' not in upper:
						upper['rotational'] = {}

					if 'rotational' not in lower:
						lower['rotational'] = {}
					#For rotations
					for i in xrange(numofangles): 						#2 is the number of samples

						if 5*i not in upper['rotational']:
							upper['rotational'][5*i] = []
						if -5*i not in upper['rotational']:
							upper['rotational'][-5*i] = []

						if 5*i not in lower['rotational']:
							lower['rotational'][5*i] = []
						if -5*i not in lower['rotational']:
							lower['rotational'][-5*i] = []

						im1 = rotateImage(upImage,5*i) 		#5 is the angle of rotation
						upper['rotational'][5*i].append(im1)

						im1 = rotateImage(lowImage,5*i)
						lower['rotational'][5*i].append(im1)

						im2 = rotateImage(upImage,-5*i)
						upper['rotational'][-5*i].append(im2)

						im2 = rotateImage(lowImage,-5*i)
						lower['rotational'][-5*i].append(im2)

					#For noise
					if 'noise' not in upper:
						upper['noise'] = {}

					if 'noise' not in lower:
						lower['noise'] = {}

					for i in xrange(numofnoise): 						#5 is the number of samples
						if i not in upper['noise']:
							upper['noise'][i] = []
						if i not in lower['noise']:
							lower['noise'][i] = []

						noise_img = sp_noise(upImage,0.05) 		#0.05 is the probability for noise
						upper['noise'][i].append(noise_img)
						noise_img = sp_noise(lowImage,0.05)
						lower['noise'][i].append(noise_img)
					

				classDict[dirn.lower()][person]['upper'] = upper 			#so upper contains images in order; has 2 levels in case of original. One for 'original' and then the images themselves, else there are 3 levels 1 for rotational/noise and another for wht type of rotationl (angle) or noise (number) and then the actual images.
				classDict[dirn.lower()][person]['lower'] = lower

		print 'The number of classes already there in classDict is',len(classDict.keys())
		if len(classDict.keys()) == numofclasses:
			break

	if len(classDict.keys()) == numofclasses:
			break
'''
for label in classDict:
	if not os.path.exists('./output/'+label): os.makedirs('./output/'+label)
	for i in xrange(len(classDict[label])):
		cv2.imwrite('./output/'+label+'/'+str(i)+'.png',classDict[label][i])
'''
print 'Writing it to pickle'
writePickle(classDict,"/net/ht140/payam-hadar/DL/classDict.pickle")
print 'Written to classDict.pickle. Execution ended...'
