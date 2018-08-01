We ran the code from the ICML18 and AnoGAN papers to compare our approach to theirs. We used the multi-class setting from our paper, namely, using MNIST and CIFAR-10 as normal data and other image datasets as anomalous data sets. We also experimented with the simpler one-class setting in ICML18, i.e., using just one class from MNIST (or from CIFAR-10) as normal data.

The main changes in the Deep-SVDD code are in src/baseline.py, src/datasets/mnist.py, src/datasets/cifar10.py, src/datasets/preprocessing.py

The main changes in the AnoGAN code are anogan.py, main.py, svhn_data.py and test.py

We will merge these codes with ours once the paper is accepted.
