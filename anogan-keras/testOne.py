from __future__ import print_function
import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist,cifar10
import argparse
import anogan

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=14)
parser.add_argument('--label_idx', type=int, default=7)
parser.add_argument('--mode', type=str, default='test', help='train, test')
args = parser.parse_args()

def anomaly_detection(test_img, g=None, d=None):
    model = anogan.anomaly_detector(g=g, d=d)
    ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 32, 32, 3), iterations=500, d=d)

    # anomaly area, 255 normalization
    np_residual = test_img.reshape(32,32,3) - similar_img.reshape(32,32,3)
    np_residual = (np_residual + 2)/4

    np_residual = (255*np_residual).astype(np.uint8)
    original_x = (test_img.reshape(32,32,3)*127.5+127.5).astype(np.uint8)
    similar_x = (similar_img.reshape(32,32,3)*127.5+127.5).astype(np.uint8)

    # original_x_color = cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)
    # residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_JET)
    # show = cv2.addWeighted(original_x_color, 0.3, residual_color, 0.7, 0.)

    return ano_score, original_x, similar_x

(X_train, y_train), (X_test, y_test) =  cifar10.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5
y_train = y_train[:,0]
y_test = y_test[:,0]

idx_normal = np.any(y_train[..., None] == np.array([0])[None, ...], axis=1)
X_train = X_train[idx_normal]

test_img = X_train[1]
start = cv2.getTickCount()
score, qurey, pred = anomaly_detection(test_img)
time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
print ('%.2f'%score, '%.2fms'%time)

plt.figure(1, figsize=(3, 3))
plt.title('query image')
plt.imshow(qurey.reshape(32,32,3))
plt.show()

print("anomaly score : ", score)
plt.figure(2, figsize=(3, 3))
plt.title('generated similar image')
plt.imshow(pred.reshape(32,32,3))
plt.show()
