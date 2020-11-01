# Project 3 – Week 1
# 1. Install the Theano library
# http://deeplearning.net/software/thano/
 
$ python -m pip install Theano
$ python -m pip install --upgrade --no-deps theano
	nosetests theano
 
 
# 2. CNN training is intensive and time consuming and can be GREATLY (1000x)
#    accelerated using GPUs.
import theano
 
GPU = True
if GPU:
   print("Trying to run under a GPU.  If this is not desired, then modify "+\
       "network3.py\nto set the GPU flag to False.")
   try: theano.config.device = 'gpu'
   except: pass # it's already set
   theano.config.floatX = 'float32'
else:
   print("Running with a CPU.  If this is not desired, then the modify "+\
   "network3.py to set\nthe GPU flag to True.")
 
 
# 3. Alternatively, you may create a free AWS amazon account and get $70+ worth of 
#    credit to run your code on an EC2 G2 instance.
#    http://aws.amazon.com/ec2/instance-types/
 
 
# 4. Import the MNIST data and CNN library:
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftMaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
 
 
# 5. Create a fully connected net with 784 input, 100 output, 10 softmax layer,
#    and a minibatch of 10.
 
net = Network([
   FullyConnectedLayer(n_in=784, n_out=100),
   SoftmaxLayer(n_in=100, n_out=10)], 10)
 
 
# 6. Train using SGD for 60+ epoch, using a learning rate = 0.1:
# Note the testing accuracy. This will be a baseline accuracy to compare CNN
 
net.SGD(training_data, 60, 10, 0.1, validation_data, test_data)
 
# 7. Create a CNN that takes in the 28x28 MNIST images,
#    uses 20 convolutional layers, each layer is created by 5x5 shared
#    filter with a stride of 1, applies a 2x2 max pooling,
#    pooling layers is fully connected to 100 sigmoid neurons,
#    followed by 10 softmax naurons
 
mini_batch_size = 10
 
net = Network([
   ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                 filter_shape=(20, 1, 5, 5), poolsize=(2, 2)),
   FullyConnectedLayer(n_in=20*12*12, n_out=100),
   SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
 
 
# 8. Train using SGD for 60+ epochusing learning rate=0.1
#   Note the performance accuracy and the number of epoch to reach baseline.
 
net.SGD(training_data, 60, 10, 0.1, validation_data, test_data)
 
# 9. Modify 7 by adding convolutional layers following pooling layers;
#    40 convolutional layers, 5x5 filters, stride of 20, pool 2x2
 
net = Network([
     ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                   filter_shape=(20, 1, 5, 5), poolsize=(2, 2)),
     ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                   filter_shape=(40, 20, 5, 5), poolsize=(2, 2)),
     FullyConnectedLayer(n_in=40*4*4, n_out=100),
     SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
 
 
# Retrain using the specs in 6. Note epochs is takes to reach baseline
 
net.SGD(training_data, 60, 10, 0.1, validation_data, test_data)
 
# 10. Modify the network in 9 to use RELU activation,
# use L2 regulariztion 𝜆=0.1, learning rate of 0.03
# Keep the softmax output.
 
net = Network([
   ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                 filter_shape=(20, 1, 5, 5),
                 poolsize=(2, 2),
                 activation_fn=ReLU),
   ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                 filter_shape=(40, 20, 5, 5),
                 poolsize=(2, 2),
                 activation_fn=ReLU),
   FullyConnectedLayer(n_in=40*4*4, n_out=100, activation_fn=ReLU),
   SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
 
 
# Retrain the network using SGD:
 
net.SGD(training_data, 60, mini_batch_size, 0.03,
       validation_data, test_data, lmbda=0.1)
 
# Note the improvement in accuracy

# 11. Modify the network in 10 by using two-1000-neuron hidden layers
# before softmax. Retain the network and note the improvement in
# accuracy and how any epochs does it take to reach the baseline.
 
net = Network([
   ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                 filter_shape=(20, 1, 5, 5),
                 poolsize=(2, 2),
                 activation_fn=ReLU),
   ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                 filter_shape=(40, 20, 5, 5),
                 poolsize=(2, 2),
                 activation_fn=ReLU),
   FullyConnectedLayer(
                 n_in=40*4*4, n_out=1000, activation_fn=ReLU),
   FullyConnectedLayer(
                 n_in=1000, n_out=1000, activation_fn=ReLU),
   SoftmaxLayer(n_in=1000, n_out=10)],
   mini_batch_size)
 
net.SGD(training_data, 60, mini_batch_size, 0.03,
       validation_data, test_data, lmbda=0.1)