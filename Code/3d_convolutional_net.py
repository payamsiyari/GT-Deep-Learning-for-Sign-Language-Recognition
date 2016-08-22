import os
import sys
import time

import theano
from theano import tensor as T
floatX = theano.config.floatX
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import scipy.stats as ss
import cPickle as pickle
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from generateInput import returnImages
#from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import theano.tensor.nnet.conv3d2d

srng = RandomStreams()

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def unpickle(filename):
	f = open(filename,"rb") 
	heroes = pickle.load(f)
	return heroes

def floatXX(X):
    return np.asarray(X, dtype=floatX)

def init_weights(shape):
    return theano.shared(floatXX(np.random.randn(*shape) * 0.01), borrow = True)

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def shared_dataset(data_x, data_y, borrow=True):
    
    shared_x = theano.shared(data_x,borrow=True)
    shared_y = theano.shared(data_y,borrow=True)
	
    return [shared_x, T.cast(shared_y, 'int32')]

def model(X, w, w2, w3, w35, w4, p_drop_conv, p_drop_hidden):

    #batch_size, in_time1, ip_kernel, ip_row1, ip_col1 = X.shape
    #op_kernel1, k_time1, ip_kernel1, conv11, conv12 = w.shape

    #l1a = T.nnet.conv3d2d.conv3d(X, w, signals_shape=X.shape, filters_shape=w.shape, border_mode='valid')
    l1a = T.nnet.conv3d2d.conv3d(X, w, border_mode='valid')
    l1b = rectify(l1a)
    l1 = max_pool_2d(l1b, (2, 2))
    #l1 = dropout(l1, p_drop_conv)
    
    #op_row1 = (ip_row1 - conv11 + 1)/2
    #op_col1 = (ip_col1 - conv12 + 1)/2
    #op_time1 = k_time1 - in_time1 + 1

    #image_shape2 = (batch_size, op_time1, op_kernel1, op_row1, op_col1)
    #op_kernel2, k_time2, ip_kernel1, conv21, conv22 = w2.shape

    #l2a = T.nnet.conv3d2d.conv3d(l1, w2, signals_shape=image_shape2, filters_shape=w2.shape)
    l2a = T.nnet.conv3d2d.conv3d(l1, w2)
    l2b = rectify(l2a)
    l2 = max_pool_2d(l2b, (2, 2))
    #l2 = dropout(l2, p_drop_conv)

    #op_row2 = (op_row1 - conv21 + 1)/2
    #op_col2 = (op_col1 - conv22 + 1)/2
    #op_time2 = op_time1 - k_time2 + 1

    #image_shape3 = (batch_size, op_time2, op_kernel2, op_row2, op_col2)
    #op_kernel3, k_time3, ip_kernel2, conv31, conv32 = w3.shape

    #l3a = T.nnet.conv3d2d.conv3d(l2, w3, signals_shape=image_shape3, filters_shape=w3.shape)
    l3a = T.nnet.conv3d2d.conv3d(l2, w3)
    l3b = rectify(l3a)
    l3 = max_pool_2d(l3b, (2, 2))
    #l3 = dropout(l3, p_drop_conv)

    #op_row3 = (op_row2 - conv31 + 1)/2
    #op_col3 = (op_col2 - conv32 + 1)/2
    #op_time3 = op_time2 - k_time3 + 1

    # flatten the time dimension
    l35_ip = l3[:,0,:,:,:]
    
    l35a = rectify(conv2d(l35_ip, w35))
    l35b = max_pool_2d(l35a, (2, 2))
    l35 = T.flatten(l35b, outdim=2)
    #l35 = dropout(l35, p_drop_conv)
    
    l4 = rectify(T.dot(l35, w4))
    #l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l35, l4, pyx

batch = 5000
classDict = unpickle("/net/ht140/payam-hadar/DL/classDictSkip4.pickle")
#classDict = unpickle("./classDictSkip4.pickle")
images, labels = returnImages(batch,4,200,200,classDict)
images=images.reshape((batch,4,1,200,200))

batch_size=100

image_size = images.shape
label_size = labels.shape

train_set_x_np = np.array(images[0:int(0.8*image_size[0]), :, :, :, :],dtype=floatX)
valid_set_x_np = np.array(images[int(0.8*image_size[0]):int(0.9*image_size[0]), :, :, :, :],dtype=floatX)
test_set_x_np = np.array(images[int(0.9*image_size[0]):image_size[0], :, :, :, :],dtype=floatX)

print train_set_x_np.shape

train_set_y_np = np.asarray(labels[0:int(0.8*image_size[0])],dtype=floatX)
valid_set_y_np = np.asarray(labels[int(0.8*image_size[0]):int(0.9*image_size[0])],dtype=floatX)
test_set_y_np = np.asarray(labels[int(0.9*image_size[0]):image_size[0]],dtype=floatX)

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x_np.shape[0]/batch_size
n_valid_batches = valid_set_x_np.shape[0]/batch_size
n_test_batches = test_set_x_np.shape[0]/batch_size

train_set_x, train_set_y = shared_dataset(train_set_x_np, train_set_y_np)
valid_set_x, valid_set_y = shared_dataset(valid_set_x_np, valid_set_y_np)
test_set_x, test_set_y = shared_dataset(test_set_x_np, test_set_y_np)

#trX, teX, trY, teY = mnist(onehot=True)
trX = train_set_x
teX = test_set_x
trY = train_set_y
teY = test_set_y

#trX = trX.reshape(-1, 1, 28, 28)
#teX = teX.reshape(-1, 1, 28, 28)

# start-snippet-1
dtensor5 = T.TensorType('float64', (False,)*5)
X = dtensor5('x')   # the data is presented as 5D Tensor
Y = T.fmatrix('y')

w = init_weights((32, 2, 1, 5, 5))
w2 = init_weights((64, 2, 32, 5, 5))
w3 = init_weights((96, 2, 64, 3, 3))
w35 = init_weights((128, 96, 3, 3))
w4 = init_weights((128 * 11 * 11, 1024))
w_o = init_weights((1024, 9))

#noise_l1, noise_l2, noise_l3, noise_l35, noise_l4, noise_py_x = model(X, w, w2, w3, w35, w4, 0.2, 0.5)
l1, l2, l3, l35, l4, py_x = model(X, w, w2, w3, w35, w4, 0., 0.)
y_x = T.argmax(py_x, axis=1)

softMaxOut = py_x
#cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w, w2, w3, w35, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=[cost, softMaxOut], updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=[y_x, softMaxOut], allow_input_downcast=True)

print "Starting Training..."

train_set_y_np = one_hot([int(i) for i in train_set_y_np],9)
test_set_y_np = one_hot([int(i) for i in test_set_y_np],9)
#print "sizeOf", train_set_y_np.shape
start_time = time.clock()
bestAcc = np.inf
bestAccTop3 = np.inf
bestAccTop5 = np.inf
for i in range(50):
    print "Epoch " + str(i) + " started..."
    epoch_start_time = time.clock()
    for start, end in zip(range(0, len(train_set_x_np), 100), range(100, len(train_set_x_np), 100)):
        print "Epoch " + str(i) + ", Batch with starting index " + str(start) + " started..."
        cost, softMaxOut = train(train_set_x_np[start:end], train_set_y_np[start:end])
        print "Cost = " + str(cost)
        print "SofMaxOut = ", softMaxOut[0]
    #print test_set_y_np
    #print np.argmax(test_set_y_np,axis=1)
    #print predict(test_set_x_np[0:2])
    pred, softMaxOut = predict(test_set_x_np[0:2])
    print "SofMaxOutTest = ", softMaxOut
    epochAcc = np.mean(np.argmax(test_set_y_np[0:2],axis=1) == pred)
    #ranking = softMaxOut.shape[1] - np.argsort(softMaxOut)
    #print ranking
    #top3EpochAcc = np.count_nonzero( ranking[:, np.argmax(test_set_y_np[0:2], axis = 1)] <= 2)
    #top5EpochAcc = np.count_nonzerp( ranking[:, np.argmax(test_set_y_np[0:2], axis = 1)] <= 4)
    #print "Prediction", np.argmax(test_set_y_np[0:2], axis=1)
    #print "tops",top3EpochAcc,top3EpochAcc
    #print "Epoch " + str(i) + " accuracy:", epochAcc
    matchedTop3 = 0
    matchedTop5 = 0
    softMaxOutSorted = np.argsort(softMaxOut)
    print "softMaxOutSorted = ", softMaxOutSorted
    print "Labels:",np.argmax(test_set_y_np,axis=1)
    for rowIndex in xrange(softMaxOutSorted.shape[0]):
        matchedTop3 += int(np.where(softMaxOutSorted[rowIndex,:] == np.argmax(test_set_y_np[rowIndex,:]))[0][0]>=softMaxOutSorted.shape[1]-3)
        matchedTop5 += int(np.where(softMaxOutSorted[rowIndex,:] == np.argmax(test_set_y_np[rowIndex,:]))[0][0]>=softMaxOutSorted.shape[1]-5)
    epochAccTop3 = matchedTop3/float(softMaxOutSorted.shape[0])
    epochAccTop5 = matchedTop3/float(softMaxOutSorted.shape[0])
    if epochAcc < bestAcc:
        bestAcc = epochAcc
    if epochAccTop3 < bestAccTop3:
        bestAccTop3 = epochAccTop3
    if epochAccTop5 < bestAccTop5:
        bestAccTop5 = epochAccTop5
    print "Epoch " + str(i) + " accuracies:"
    print "\t epochAcc",epochAcc
    print "\t epochAccTop3",epochAccTop3,"bestSoFar:",bestAccTop3
    print "\t epochAccTop5",epochAccTop5,"bestSoFar:",bestAccTop5
    epoch_end_time = time.clock()
    print "Epoch Time:" + str(epoch_end_time - epoch_start_time) + "seconds"
    print "Saving weights..."
    f = open('/net/ht140/payam-hadar/DL/saved_weight_w_3D', 'wb')
    f1 = open('/net/ht140/payam-hadar/DL/saved_weight_w2_3D', 'wb')
    f2 = open('/net/ht140/payam-hadar/DL/saved_weight_w3_3D', 'wb')
    f3 = open('/net/ht140/payam-hadar/DL/saved_weight_w35_3D', 'wb')
    f35 = open('/net/ht140/payam-hadar/DL/saved_weight_w4_3D', 'wb')
    f4 = open('/net/ht140/payam-hadar/DL/saved_weight_w_o_3D', 'wb')
    pickle.dump(w, f)
    pickle.dump(w2, f1)
    pickle.dump(w3, f2)
    pickle.dump(w35, f3)
    pickle.dump(w4, f35)
    pickle.dump(w_o, f4)
    f.close()
    f1.close()
    f2.close()
    f3.close()
    f35.close()
    f4.close()
    print "done."
end_time = time.clock()
print "Best accuracy:", bestAcc
print "Total Time:",((end_time - start_time) / 60.)
