import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from createGIF import createGIF
import pickle


def visualize(convA):
	if convA == None:
		return 

	#imgY = 10

	for i in xrange(convA.shape[1]):
		print i
		temp = convA[:,i,:,:,:]
		print temp.shape
		#k = convA.shape[1] * convA.shape[0]
		#j = int(round(float(k) / imgY))
		gs1 = gridspec.GridSpec(temp.shape[0],temp.shape[1])
		gs1.update(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.3)
		dim = max(temp.shape[0],temp.shape[1])
		plt.figure(figsize=(dim,dim))
		for x in xrange(temp.shape[0]):
			for y in xrange(temp.shape[1]):
				w = temp[x,y,:,:]
				ax = plt.subplot(gs1[x,y])
				ax.imshow(w,cmap=plt.cm.gist_yarg,interpolation='nearest',aspect='auto')
				ax.axis('off')
				

		plt.axis('off')
		plt.tick_params(\
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom='off',      # ticks along the bottom edge are off
			top='off',         # ticks along the top edge are off
			labelbottom='off')
		plt.tick_params(\
			axis='y',          # changes apply to the y-axis
			which='both',      # both major and minor ticks are affected
			left='off', 
			right='off',    # ticks along the top edge are off
			labelleft='off')




		plt.savefig('./tempImages/test_fig_' + str(i) + '.png', dpi = 100)
		plt.close('all')
	createGIF()




'''
[inpchannel,outchannel,time,row,column]

[x,y,time,a,b] - generates an image of x,y dimensions where each row has multiple axb weights]
'''

weights0 = pickle.load(open('/net/ht140/payam-hadar/DL/saved_weight_w_3D','rb'))
#weights1 = pickle.load(open('saved_weight_layer1','rb'))
print weights0.shape
#print weights1.shape
visualize(weights0.get_value())
