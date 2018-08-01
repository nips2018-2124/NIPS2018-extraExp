from __future__ import print_function

import matplotlib
matplotlib.use('Qt5Agg')
import torchvision.transforms as transforms
import torch
import torchvision

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist, cifar10, cifar100
import argparse
import anogan
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=14)
parser.add_argument('--label_idx', type=int, default=7)
parser.add_argument('--mode', type=str, default='train', help='train, test')
args = parser.parse_args()

def converter_to_small(x):
    #x has shape (batch, width, height, channels)
    gray_images =  (0.21 * x[:,:,:,:1]) + (0.72 * x[:,:,:,1:2]) + (0.07 * x[:,:,:,-1:])
    resized_images = gray_images[:,2:30,2:30,:]
    return resized_images

### 0. prepare data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# X_train = converter_to_small(X_train)
y_train = y_train[:,0]
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

print ('train shape:', X_train.shape)
print('train label shape', y_train.shape)
idx_normal = np.any(y_train[..., None] == np.array([0])[None, ...], axis=1)
X_train = X_train[idx_normal]

# X_train = X_train[:,:,:,None]
# X_test = X_test[:,:,:,None]

# X_train = X_train[y_train==0]
# X_test = X_test[y_test==0]
# print(np.where(y_train == 0))

# perm_test = np.random.permutation(len(X_train))
# X_train = X_train[perm_test]
# X_train = X_train[:10000]

print ('train shape:', X_train.shape)

### 1. train generator & discriminator
if args.mode == 'train':
    Model_d, Model_g = anogan.train(64, X_train)



### compute outlier scores on CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
y_train = y_train[:,0]
y_test = y_test[:,0]

idx_normal = np.any(y_train[..., None] == np.array([0])[None, ...], axis=1)
X_train = X_train[idx_normal]
perm_test = np.random.permutation(len(X_train))
X_train = X_train[perm_test]

## 3. other class anomaly detection
test_imgs = X_train[0:500]

score_for_2 = []
# test_img = np.random.uniform(-1,1, (28,28,1))
model = anogan.anomaly_detector(g=None, d=None)
for i in range(np.shape(test_imgs)[0]):
    # start = cv2.getTickCount()
    score = anogan.compute_anomaly_score(model, test_imgs[i].reshape(1, 32, 32, 3), iterations=500, d=None)
    # time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print ('%d label, %d : done'%(1, i), '%.2f'%score)
    score_for_2.append(score)

score_for_2 = np.sort(score_for_2)[::-1]
print(score_for_2)
import pickle
pickle.dump(score_for_2, open( "cifar10_score.pickle", "wb" ) )

### compute outlier scores on CIFAR-100
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
# X_train = X_train[:,:,:,None]
# X_test = X_test[:,:,:,None]
perm_test = np.random.permutation(len(X_train))
X_train = X_train[perm_test]

test_imgs = X_train[0:200]
score_for_2 = []
# test_img = np.random.uniform(-1,1, (28,28,1))
model = anogan.anomaly_detector(g=None, d=None)
for i in range(np.shape(test_imgs)[0]):
    # start = cv2.getTickCount()
    score = anogan.compute_anomaly_score(model, test_imgs[i].reshape(1, 32, 32, 3), iterations=500, d=None)
    # time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print ('%d label, %d : done'%(1, i), '%.2f'%score)
    score_for_2.append(score)
score_for_2 = np.sort(score_for_2)[::-1]
print(score_for_2)
import pickle
pickle.dump(score_for_2, open( "cifar100_score.pickle", "wb" ) )


### compute outlier scores on SVHN
import svhn_data
(X_train, y_train), (X_test, y_test) = svhn_data.load_svhn_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
# X_train = X_train[:,:,:,None]
# X_test = X_test[:,:,:,None]
perm_test = np.random.permutation(len(X_train))
X_train = X_train[perm_test]

test_imgs = X_train[0:200]
score_for_2 = []
# test_img = np.random.uniform(-1,1, (28,28,1))
model = anogan.anomaly_detector(g=None, d=None)
for i in range(np.shape(test_imgs)[0]):
    # start = cv2.getTickCount()
    score = anogan.compute_anomaly_score(model, test_imgs[i].reshape(1, 32, 32, 3), iterations=500, d=None)
    # time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print ('%d label, %d : done'%(1, i), '%.2f'%score)
    score_for_2.append(score)
score_for_2 = np.sort(score_for_2)[::-1]
print(score_for_2)
import pickle
pickle.dump(score_for_2, open( "svhn_score.pickle", "wb" ) )


### compute outlier scores on MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5


def convert_to_large(X):
    X_new  =  np.zeros((X.shape[0],32,32,3))
    X_new[:,2:30,2:30,0] = X
    X_new[:,2:30,2:30,1] = X
    X_new[:,2:30,2:30,2] = X
    return X_new
X_train = convert_to_large(X_train)
X_test = convert_to_large(X_test)

test_imgs = X_test[0:200]
score_for_2 = []
# test_img = np.random.uniform(-1,1, (28,28,1))
model = anogan.anomaly_detector(g=None, d=None)
for i in range(np.shape(test_imgs)[0]):
    start = cv2.getTickCount()
    score = anogan.compute_anomaly_score(model, test_imgs[i].reshape(1, 32, 32, 3), iterations=100, d=None)
    time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
    print ('%d label, %d : done'%(1, i), '%.2f'%score, '%.2fms'%time)
    score_for_2.append(score)
score_for_2 = np.sort(score_for_2)[::-1]
print(score_for_2)
import pickle
pickle.dump(score_for_2, open( "mnist_score.pickle", "wb" ) )

