from images2gif import writeGif
from PIL import Image
import os
from pylab import imread, imshow, gray, mean
import pickle


def createGIF():


	file_names = sorted((x for x in os.listdir('./tempImages') if x.endswith('.png')))
	images = [Image.open('./tempImages/'+fn) for fn in file_names]

	filename = "./tempImages/result.gif"
	writeGif(filename, images, duration=0.2)



