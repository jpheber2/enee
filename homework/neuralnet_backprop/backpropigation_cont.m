# Project 2 - Week 2
# 1. Change the initialization scheme so that it does Xavier initialization
 
   def default_weight_initializer(self):
 
 self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
       self.weights = [np.random.randn(y, x) * np.sqrt(6.0/(x+y))
                       for x, y in zip(self.sizes[:-1], self.sizes[1:])]
 
 
# 2. Change the learning so that is follows Adagrad
# Update weights
   def update_mini_batch(self, mini_batch, eta, lmbda, n):
 
       nabla_b = [np.zeros(b.shape) for b in self.biases]
       nabla_w = [np.zeros(w.shape) for w in self.weights]
       for x, y in mini_batch:
           delta_nabla_b, delta_nabla_w = self.backprop(x, y)
           nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
           nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
           
       self.weights = [w-eta*nw/((1e-6)+np.sqrt(nw*nw))
                       for w, nw in zip(self.weights, nabla_w)]
       self.biases = [b-(eta/len(mini_batch))*nb
                      for b, nb in zip(self.biases, nabla_b)]
 
 
# 3. Use 1,2 and repeat step 15 from week1
 
net = Network([784,100,10],cost=CrossEntropyCost)
net.ada(training_data, 30, 10, 0.5, evaluation_data=validation_data,
       lmbda=5.0, monitor_evaluation_accuracy=True)
 
# 5. Download the notMNIST dataset 
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.

from __future__ import print_function
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
 
# Config the matplotlib backend as plotting inline in IPython
%matplotlib inline
 
url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.' # Change me to store data elsewhere
 
def download_progress_hook(count, blockSize, totalSize):
 """A hook to report the progress of a download. This is mostly intended for users with
 slow internet connections. Reports every 5% change in download progress.
 """
 global last_percent_reported
 percent = int(count * blockSize * 100 / totalSize)
 
 if last_percent_reported != percent:
   if percent % 5 == 0:
     sys.stdout.write("%s%%" % percent)
     sys.stdout.flush()
   else:
     sys.stdout.write(".")
     sys.stdout.flush()
     
   last_percent_reported = percent
       
def maybe_download(filename, expected_bytes, force=False):
 """Download a file if not present, and make sure it's the right size."""
 dest_filename = os.path.join(data_root, filename)
 if force or not os.path.exists(dest_filename):
   print('Attempting to download:', filename)
   filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
   print('\nDownload Complete!')
 statinfo = os.stat(dest_filename)
 if statinfo.st_size == expected_bytes:
   print('Found and verified', dest_filename)
 else:
   raise Exception(
     'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
 return dest_filename
 
train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
 
Found and verified ./notMNIST_large.tar.gz
Found and verified ./notMNIST_small.tar.gz
 
num_classes = 10
np.random.seed(133)
 
def maybe_extract(filename, force=False):
 root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
 if os.path.isdir(root) and not force:
   # You may override by setting force=True.
   print('%s already present - Skipping extraction of %s.' % (root, filename))
 else:
   print('Extracting data for %s. This may take a while. Please wait.' % root)
   tar = tarfile.open(filename)
   sys.stdout.flush()
   tar.extractall(data_root)
   tar.close()
 data_folders = [
   os.path.join(root, d) for d in sorted(os.listdir(root))
   if os.path.isdir(os.path.join(root, d))]
 if len(data_folders) != num_classes:
   raise Exception(
     'Expected %d folders, one per class. Found %d instead.' % (
       num_classes, len(data_folders)))
 print(data_folders)
 return data_folders
 
train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)
 
./notMNIST_large already present - Skipping extraction of ./notMNIST_large.tar.gz. ['./notMNIST_large/A', './notMNIST_large/B', './notMNIST_large/C', './notMNIST_large/D', './notMNIST_large/E', './notMNIST_large/F', './notMNIST_large/G', './notMNIST_large/H', './notMNIST_large/I', './notMNIST_large/J'] ./notMNIST_small already present - Skipping extraction of ./notMNIST_small.tar.gz. ['./notMNIST_small/A', './notMNIST_small/B', './notMNIST_small/C', './notMNIST_small/D', './notMNIST_small/E', './notMNIST_small/F', './notMNIST_small/G', './notMNIST_small/H', './notMNIST_small/I', './notMNIST_small/J']
 
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
 
def load_letter(folder, min_num_images):
 """Load the data for a single letter label."""
 image_files = os.listdir(folder)
 dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                        dtype=np.float32)
 print(folder)
 num_images = 0
 for image in image_files:
   image_file = os.path.join(folder, image)
   try:
     image_data = (imageio.imread(image_file).astype(float) -
                   pixel_depth / 2) / pixel_depth
     if image_data.shape != (image_size, image_size):
       raise Exception('Unexpected image shape: %s' % str(image_data.shape))
     dataset[num_images, :, :] = image_data
     num_images = num_images + 1
   except (IOError, ValueError) as e:
     print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
   
 dataset = dataset[0:num_images, :, :]
 if num_images < min_num_images:
   raise Exception('Many fewer images than expected: %d < %d' %
                   (num_images, min_num_images))
   
 print('Full dataset tensor:', dataset.shape)
 print('Mean:', np.mean(dataset))
 print('Standard deviation:', np.std(dataset))
 return dataset
       
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
 dataset_names = []
 for folder in data_folders:
   set_filename = folder + '.pickle'
   dataset_names.append(set_filename)
   if os.path.exists(set_filename) and not force:
     # You may override by setting force=True.
     print('%s already present - Skipping pickling.' % set_filename)
   else:
     print('Pickling %s.' % set_filename)
     dataset = load_letter(folder, min_num_images_per_class)
     try:
       with open(set_filename, 'wb') as f:
         pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
     except Exception as e:
       print('Unable to save data to', set_filename, ':', e)
 
 return dataset_names
 
train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)
 
def make_arrays(nb_rows, img_size):
 if nb_rows:
   dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
   labels = np.ndarray(nb_rows, dtype=np.int32)
 else:
   dataset, labels = None, None
 return dataset, labels
 
def merge_datasets(pickle_files, train_size, valid_size=0):
 num_classes = len(pickle_files)
 valid_dataset, valid_labels = make_arrays(valid_size, image_size)
 train_dataset, train_labels = make_arrays(train_size, image_size)
 vsize_per_class = valid_size // num_classes
 tsize_per_class = train_size // num_classes
   
 start_v, start_t = 0, 0
 end_v, end_t = vsize_per_class, tsize_per_class
 end_l = vsize_per_class+tsize_per_class
 for label, pickle_file in enumerate(pickle_files):      
   try:
     with open(pickle_file, 'rb') as f:
       letter_set = pickle.load(f, encoding='latin1')
       # let's shuffle the letters to have random validation and training set
       np.random.shuffle(letter_set)
       if valid_dataset is not None:
         valid_letter = letter_set[:vsize_per_class, :, :]
         valid_dataset[start_v:end_v, :, :] = valid_letter
         valid_labels[start_v:end_v] = label
         start_v += vsize_per_class
         end_v += vsize_per_class
                   
       train_letter = letter_set[vsize_per_class:end_l, :, :]
       train_dataset[start_t:end_t, :, :] = train_letter
       train_labels[start_t:end_t] = label
       start_t += tsize_per_class
       end_t += tsize_per_class
   except Exception as e:
     print('Unable to process data from', pickle_file, ':', e)
     raise
   
 return valid_dataset, valid_labels, train_dataset, train_labels
 
valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
 train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)
 
print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)
 
def randomize(dataset, labels):
 permutation = np.random.permutation(labels.shape[0])
 shuffled_dataset = dataset[permutation,:,:]
 shuffled_labels = labels[permutation]
 return shuffled_dataset, shuffled_labels
 
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
 
image_size = 28
num_labels = 10
 
print (train_labels[10])
 
def reformat(dataset, labels):
 dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
 return dataset, labels
 
def reformat_train(dataset, labels):
 dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)  
 labels = (np.arange(1,num_labels+1) == labels[:,None]).astype(np.float32)
 return dataset, labels
 
valid_dataset, valid_labels = reformat(valid_dataset,valid_labels)
train_dataset, train_labels = reformat_train(train_dataset, train_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
 
test_dataset = np.expand_dims(test_dataset, axis=2)
train_dataset = np.expand_dims(train_dataset, axis=2)
valid_dataset = np.expand_dims(valid_dataset, axis=2)
train_labels = np.expand_dims(train_labels, axis=2)
 
print (train_labels[10])
 
print ('Training set', train_dataset.shape, train_labels.shape)
print ('Validation set', valid_dataset.shape, valid_labels.shape)
print ('Test set', test_dataset.shape, test_labels.shape)
 
train_data = zip(train_dataset, train_labels)
valid_data = zip(valid_dataset, valid_labels)
testing_data = zip(test_dataset, test_labels)
 
 
# 6. Repeat 3 on the notMNIST data
 
net = Network([784,100,10],cost=CrossEntropyCost)
net.ada(train_data, 30, 10, 0.5, evaluation_data=valid_data,
       lmbda=5.0, monitor_evaluation_accuracy=True)
 
 
# 7. Modify to include RELU and softmax function, and the gradient
 
# Rectified linear unit activation function
# Max{0,Z}
 
def relu(z):
   """The ReLU function."""
   return z*(z>0)
 
def relu_prime(z):
   """Derivative of the ReLU function."""
   return 1.*(z>0)
 
 
class CrossEntropyCost(object):
   @staticmethod
   def fn(a, y):
       a = softmax(a)
       return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
 
   def softmax(z):
       n_z = z - np.max(z)
       exps = np.exp(n_z)
       return exps / np.sum(exps)
 
 
# 8. Create a network with
#   784 inputs, 4 hidden layers x 30 ReLU neurons, and 10 softmax outputs
 
net = Network([784,30,30,30,30,10],cost=CrossEntropyCost)
 
 
# 9. Change the initialization scheme into a RELU initialization scheme
 
class Network(object):
 
   def default_weight_initializer(self):
 
       self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
       self.weights = [np.random.randn(y, x) * np.sqrt(6.0/x)
                       for x, y in zip(self.sizes[:-1], self.sizes[1:])]
 
 
# 10. Process the notMNIST data
 
net = Network([784,30,30,30,30,10],cost=CrossEntropyCost)
net.ada(train_data, 30, 10, 0.5, evaluation_data=valid_data,
       lmbda=5.0, monitor_evaluation_accuracy=True)
