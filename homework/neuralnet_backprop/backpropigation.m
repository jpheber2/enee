	# Project 2 - Week 1
# 1. Start by getting the mnist data
# 2. Unzip the data and take a look at the images it contains
 
import cPickle
import gzip
import numpy as np
 
f = gzip.open('/Users/Jacques/Downloads/mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = cPickle.load(f)
 
 
train_x, train_y = training_data
 
import matplotlib.cm as cm
import matplotlib.pyplot as plt
 
 
plt.imshow(train_x[0].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()


# 3. Use mnist_loader.py Determine the size of each image, range of data,
# ...number of images in the training/testing/validation sets
 
load '/Users/Jacques/Downloads/neural-networks-and-deep-learning-master/src/mnist_loader.py'

sys.path.append('/Users/Jacques/Downloads/neural-networks-and-deep-learning-master/src/')
import mnist_loader
mnist_loader.load_data
mnist_loader.load_data_wrapper()
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# 4. Use the network.py code to create a neural network with 784 input neurons,
# ...10 output neurons and 30 hidden neurons.
 
import network
net = network.Network([784, 30, 10])

# 5. Train the network using stocastic gradient descent for 30 epochs,
#...with a mini-batch size of 10, and a learning rate of 3.0
​
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# 6. Repeat training step 4,5  but with 100 hidden neurons
​
net = network.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# 7. Repeat training step 4,5 but for learning rate of 0.001, and another for 100.0
​
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 0.001, test_data=test_data)

net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 100.0, test_data=test_data)

# 8.Use network2.py and repeat 4,5 using cross-entropy cost function,
#... learning rate of 0.5, using 10000 training data samples
 
import network2
 
random.seed(12345678)
np.random.seed(12345678)
 
net=network2.Network([784,30,10],cost=network2.CrossEntropyCost)
 
net.large_weight_initializer()
 
test_cost, test_accuracy, training_cost, training_accuracy \
=net.SGD(training_data[:1000], 400, 10, 0.5,
        evaluation_data=test_data, monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True, monitor_training_cost=True,
        monitor_training_accuracy=True)
 
 
# 9.For the 8, make plots: cost vs epoch, accuracy vs epoch
 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(200, 400), training_cost[200:400], color='#2A6EA6')
ax.set_xlim([200, 400])
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Cost on the training data')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(200, 400),
   [accuracy/100.0
   for accuracy in test_accuracy[200:400]],
   color='#2A6EA6')
ax.set_xlim([200, 400])
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Accuracy (%) on the test data')
plt.show()

# 10. Repeat 8,9 using a regularization parameter=0.1
 
net=network2.Network([784,30,10],cost=network2.CrossEntropyCost)
 
net.large_weight_initializer()
 
test_cost, test_accuracy, training_cost, training_accuracy \
=net.SGD(training_data[:1000], 400, 10, 0.5,
        evaluation_data=test_data, lmbda=0.1,
        monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
        monitor_training_cost=True, monitor_training_accuracy=True)
 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(200, 400), training_cost[200:400], color='#2A6EA6')
ax.set_xlim([200, 400])
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Cost on the training data')
plt.show()

# 11. Repeat 8,9 using a regularization parameter=5.0
 
net=network2.Network([784,30,10],cost=network2.CrossEntropyCost)
net.large_weight_initializer()
 
test_cost, test_accuracy, training_cost, training_accuracy \
=net.SGD(training_data[:1000], 400, 10, 0.5,
        evaluation_data=test_data, lmbda=5.0,
        monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
        monitor_training_cost=True, monitor_training_accuracy=True)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(200, 400),
   [accuracy/100.0
   for accuracy in test_accuracy[200:400]],
   color='#2A6EA6')
ax.set_xlim([200, 400])
ax.grid(True)
ax.set_xlabel('Epoch')
ax.set_title('Accuracy (%) on the test data')
plt.show()

# 12. Train the network with the entire training data and λ=5.0
 
net=network2.Network([784,100,10],cost=network2.CrossEntropyCost)
 
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True)

# 13. Repeat 12 for a deep network with two hidden layers, 30 neurons each
 
net=network2.Network([784,30,30,10],cost=network2.CrossEntropyCost)
 
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0,
        evaluation_data=validation_data, monitor_evaluation_accuracy=True)

# 14. Repeat 12 for a deep network with three hidden layers, 30 neurons each
 
net=network2.Network([784,30,30,30,10],cost=network2.CrossEntropyCost)
 
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0,
        evaluation_data=validation_data, monitor_evaluation_accuracy=True)

# 15. Repeat 12 for a deep network with four hidden layers, 30 neurons each
 
net=network2.Network([784,30,30,30,30,10],cost=network2.CrossEntropyCost)
 
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0,
        evaluation_data=validation_data, monitor_evaluation_accuracy=True)

# 16. Modify the network2.py by adding tanh function, and its gradient
 
define tanh(z)
​	“””The hyperbolic tangent function”””
​	return np.tanh(z)
 
define tanh_prime(z)
​	“””derivative of the tanh function”””
​	return 1.0 – np.tanh(z)**2

# 17. Repeat 12,15 for the tanh function
 
net=network2.Network([784,100,10],cost=network2.CrossEntropyCost)
 
net.SGD(training_data, 30, 10, 0.5, lmbda=5.0,
        evaluation_data=validation_data, monitor_evaluation_accuracy=True)

# NOTE LEARNING RATE 0.5 FROM STEP 12 REDUCED TO 0.1
 
net.SGD(training_data, 30, 10, 0.1, lmbda=5.0,
        evaluation_data=validation_data, monitor_evaluation_accuracy=True)

net=network2.Network([784,30,30,30,30,10],cost=network2.CrossEntropyCost)
​
net.SGD(training_data, 30, 10, 0.1, lmbda=5.0,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True)

# 18. Modify the network2.py by adding Lecun's tanh function, and its gradient
 
define lecun_tanh(z)
​	“””Lecun’s hyperbolic tangent function”””
	​return 1.7159*np.tanh(2/3*z)
 
define lecun_tanh_prime(z)
​	“””derivative of Lecun’s tanh function”””
​	return 1.14393*(1.0 – np.tanh(2/3*z)**2)

# 19. Repeat 12,15 for new tanh function
 
net=network2.Network([784,100,10],cost=network2.CrossEntropyCost)
 
net.SGD(training_data, 30, 10, 0.1, lmbda=5.0,
        evaluation_data=validation_data, monitor_evaluation_accuracy=True)
