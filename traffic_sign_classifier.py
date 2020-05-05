#!/usr/bin/env python3

import cv2
import numpy as np

# Step 0: Load The Data

# Load pickled data
import pickle

import matplotlib.pyplot as plt
from random import seed, random

from skimage import exposure
from skimage.transform import resize
from scipy.ndimage.interpolation import rotate

import sys
import math


# TODO: Fill this in based on where you saved the training and testing data

def Normalize(x):
    return (x.astype(float) - 128) / 128

def HistogramEqualize(img):
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)
training_file = "traffic-signs-data/train.p"
validation_file = "traffic-signs-data/valid.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

#Normalize inputs
X_train = Normalize(X_train)
X_validation = Normalize(X_validation)
X_test = Normalize(X_test)

# Set zero mean to inputs
X_train -= np.mean(X_train)
X_validation -= np.mean(X_validation)
X_test -= np.mean(X_test)


X_train_gray = np.zeros((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1), dtype=type(X_train[0][0][0][0]))
X_validation_gray = np.zeros((X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1), dtype=type(X_validation[0][0][0][0]))
X_test_gray = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1), dtype=type(X_test[0][0][0][0]))

for i in range(0, X_train.shape[0]):
    X_train_gray[i] = np.array(list(map(lambda x : list(map(lambda x : [x], x)), np.dot(X_train[i], [1,1,1]))))

for i in range(0, X_validation.shape[0]):
    X_validation_gray[i] = np.array(list(map(lambda x : list(map(lambda x : [x], x)), np.dot(X_validation[i], [1,1,1]))))

for i in range(0, X_test.shape[0]):
    X_test_gray[i] = np.array(list(map(lambda x : list(map(lambda x : [x], x)), np.dot(X_test[i], [1,1,1]))))

X_train_gray = HistogramEqualize(X_train_gray)
X_validation_gray = HistogramEqualize(X_validation_gray)
X_test_gray = HistogramEqualize(X_test_gray)

assert(len(X_train_gray) == len(y_train))
assert(len(X_validation_gray) == len(y_validation))
assert(len(X_test_gray) == len(y_test))

# Step 1: Dataset Summary & Exploration

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train_gray)

# TODO: Number of validation examples
n_validation = len(X_validation)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = [32,32]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = 43

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

###

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline

# Step 2: Design and Test a Model Architecture

# Pre-process the Data Set (normalization, grayscale, etc.)

from sklearn.utils import shuffle

X_train_gray, y_train = shuffle(X_train_gray, y_train)

# Model Architecture

### Define your architecture here.
### Feel free to use as many code cells as needed.

import tensorflow as tf

EPOCHS = 25
BATCH_SIZE = 42

from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x40.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 40), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(40))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x40. Output = 14x14x40.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # SOLUTION: Layer 2: Convolutional. Input = 14x14x40. Output = 10x10x70.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 40, 70), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(70))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Layer 3: Convolutional. Output = 6x6x110.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 70, 110), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(110))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b

    # SOLUTION: Activation.
    conv3 = tf.nn.relu(conv3)

    # SOLUTION: Pooling. Input = 6x6x110. Output = 3x3x110.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 3x3x110. Output = 990.
    fc0   = flatten(conv3)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 990. Output = 400.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(990, 400), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(400))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # Add drop out
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # SOLUTION: Layer 4: Fully Connected. Input = 400. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(400, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # Add drop out
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = n_classes.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

# Train, Validate and Test the Model

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
keep_prob = tf.placeholder(tf.float32) # probability to keep units

rate = 0.001
keep_probability = 0.5

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: keep_probability})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print(len(X_train_gray))
    for i in range(EPOCHS):
        X_train_gray, y_train = shuffle(X_train_gray, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_gray[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: keep_probability})
            
        validation_accuracy = evaluate(X_validation_gray, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
    
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test_gray, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))