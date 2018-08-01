import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import scipy.io as sio
import h5py

def load_images(path):
    train_images = sio.loadmat(path+'/train_32x32.mat')
    test_images = sio.loadmat(path+'/test_32x32.mat')

    return train_images, test_images


def normalize_images(images):
    imgs = images["X"]
    imgs = np.transpose(imgs, (3, 0, 1, 2))

    labels = images["y"]
    # replace label "10" with label "0"
    labels[labels == 10] = 0

    return imgs, labels


def load_svhn_data():
    train_images, test_images = load_images('./data')
    x_train, y_train = normalize_images(train_images)
    x_test, y_test = normalize_images(test_images)
    return (x_train, y_train[:,0]), (x_test, y_test[:,0])


# (X_train, y_train), (X_test, y_test) = load_svhn_data()
#
# print(X_train.shape)
# print(y_train.shape)
#
# print(X_test.shape)
# print(y_test.shape)

# print(y_test[0:5])

#
# plt.imshow(X_test[1].reshape(32,32,3))
# plt.show()
# #
# plt.imshow(X_test[2])
# plt.show()
#
# plt.imshow(X_test[3])
# plt.show()
#
# plt.imshow(X_test[4])
# plt.show()


