from datasets.preprocessing import getCIFAR10DatasetOneClass
import numpy as np
import matplotlib.pyplot as plt

CIFAR10_data_numpy = getCIFAR10DatasetOneClass()
plt.imshow(np.transpose(CIFAR10_data_numpy[0], (1, 2, 0)))
plt.show()

plt.imshow(np.transpose(CIFAR10_data_numpy[1], (1, 2, 0)))
plt.show()

plt.imshow(np.transpose(CIFAR10_data_numpy[2], (1, 2, 0)))
plt.show()

plt.imshow(np.transpose(CIFAR10_data_numpy[3], (1, 2, 0)))
plt.show()