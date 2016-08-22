#import cv2
#from cv2 import cv
#from cv2.cv import *
import numpy as np 
import random
import os
import cPickle as pickle
from random import randint

def unpickle(filename):
	f = open(filename,"rb") 
	heroes = pickle.load(f)
	return heroes

def writePickle(struct, filename):
	file1 = open(filename,"wb") 			
	pickle.dump(struct,file1)
	file1.close()

def returnImages(batchSize,in_time,width,height, classDict):
	#classDict = unpickle("classDict.pickle")
	images = np.zeros((batchSize,in_time,1,width,height))
	classes = np.zeros(batchSize)
	f = open("classes.txt","w")
	for i in xrange(len(classDict.keys())):
		f.write(str(i)+':'+str(classDict.keys()[i])+'\n')
	f.close()
	for i in xrange(batchSize):
         labelnum = randint(0,len(classDict.keys())-1) 			#randomly generate a label
         label = classDict.keys()[labelnum] 						#Chose a label ie word
         print "LABEL",label
         print "classDictKEYS",classDict[label].keys()
         person = randint(0,len(classDict[label].keys())-1) 		#randomly generate a person's name
         person = classDict[label].keys()[person]
         #typeOfImage = randint(0,1) 								#upper or lower
         t = 'upper'
         #if typeOfImage == 1:
             #t = 'lower'
    	#	mode = 'original' #['original','rotational','noise'] 		If all the modes are enabled
         m = randint(0,2)
         if m == 0:
             mode = 'original'
         elif m == 1:
             mode = 'rotational'
         else:
             mode = 'noise'
         print label,person,t,mode
         if mode == 'original':
             startpt = randint(0,len(classDict[label][person][t][mode])-in_time-1) 				#randomly generate a start point
             for j in xrange(startpt,startpt+in_time):
                 images[i,j-startpt,:,:,:] = classDict[label][person][t][mode][j]
                 #cv2.imshow("cropped", classDict[label][person][t][mode][j])
                 #cv2.waitKey(0)
         else:
             typeOfRotationOrNoise = randint(0,len(classDict[label][person][t][mode])-1)
             typeOfRotationOrNoise = classDict[label][person][t][mode].keys()[typeOfRotationOrNoise]
             startpt = randint(0,len(classDict[label][person][t][mode][typeOfRotationOrNoise])-in_time-1) 				#randomly generate a start point
             for j in xrange(startpt,startpt+in_time):
                 images[i,j-startpt,:,:,:] = classDict[label][person][t][mode][typeOfRotationOrNoise][j]
                 #cv2.imshow("cropped", classDict[label][person][t][mode][typeOfRotationOrNoise][j])
                 #cv2.waitKey(0)
         #cv2.destroyAllWindows()
         classes[i] = labelnum
         
	return [images,classes]
     
if __name__ == "__main__":
    classDict = unpickle("/net/ht140/payam-hadar/DL/classDict.pickle")
    x = returnImages(20,1,200,200,classDict) 			#batchSIze,inTime,width of image, height of image - set this in image_processor.py as well.
    print x[0].shape
    print x[1].shape