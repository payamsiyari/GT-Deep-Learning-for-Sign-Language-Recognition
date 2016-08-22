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
    l1a = rectify(conv2d(X, w, border_mode='full'))
    #print "l1a",l1a.type
    #print "l1a",l1a.shape.eval()
    l1 = max_pool_2d(l1a, (2, 2))
    #print "l1",l1.get_value().shape
    #l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    #print "l2a",l2a.get_value().shape
    l2 = max_pool_2d(l2a, (2, 2))
    #print "l2",l2.get_value().shape
    #l2 = dropout(l2, p_drop_conv)

    l3 = rectify(conv2d(l2, w3))
    #print "l3",l3.get_value().shape
    #l3 = max_pool_2d(l3a, (1, 1))
    #l3 = dropout(l3, p_drop_conv)

    l35a = rectify(conv2d(l3, w35))
    #print "l35a",l35a.get_value().shape
    l35b = max_pool_2d(l35a, (2, 2))
    #print "l35b",l35b.get_value().shape
    l35 = T.flatten(l35b, outdim=2)
    #print "l35",l35.get_value().shape
    #l35 = dropout(l35, p_drop_conv)
    
    l4 = rectify(T.dot(l35, w4))
    #print "l4",l4.get_value().shape
    #l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l35, l4, pyx

batch = 5000
classDict = unpickle("/net/ht140/payam-hadar/DL/classDictSkip4.pickle")
#classDict = unpickle("classDict.pickle")
images, labels = returnImages(batch,1,200,200,classDict)
images=images.reshape((batch,1,200,200))

batch_size=100

image_size = images.shape
label_size = labels.shape

train_set_x_np = np.array(images[0:int(0.8*image_size[0]), :, :, :],dtype=floatX)
valid_set_x_np = np.array(images[int(0.8*image_size[0]):int(0.9*image_size[0]), :, :, :],dtype=floatX)
test_set_x_np = np.array(images[int(0.9*image_size[0]):image_size[0], :, :, :],dtype=floatX)

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

X = T.dtensor4()
Y = T.fmatrix()

w = init_weights((32, 1, 5, 5))
w2 = init_weights((64, 32, 5, 5))
w3 = init_weights((96, 64, 3, 3))
w35 = init_weights((128, 96, 3, 3))
w4 = init_weights((128 * 23 * 23, 1024))
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
    pred, softMaxOut = predict(test_set_x_np)
    print "SofMaxOutTest = ", softMaxOut
    epochAcc = np.mean(np.argmax(test_set_y_np,axis=1) == pred)
    #print softMaxOut.shape
    #print np.argsort(softMaxOut)
    #ranking = np.argsort(softMaxOut)[:,::-1]
    #print "Ranking", ranking
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
    f = open('/net/ht140/payam-hadar/DL/saved_weight_w_2D', 'wb')
    f1 = open('/net/ht140/payam-hadar/DL/saved_weight_w2_2D', 'wb')
    f2 = open('/net/ht140/payam-hadar/DL/saved_weight_w3_2D', 'wb')
    f3 = open('/net/ht140/payam-hadar/DL/saved_weight_w35_2D', 'wb')
    f35 = open('/net/ht140/payam-hadar/DL/saved_weight_w4_2D', 'wb')
    f4 = open('/net/ht140/payam-hadar/DL/saved_weight_w_o_2D', 'wb')
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