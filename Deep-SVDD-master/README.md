###  We ran the code from the ICML18 paper to compare our approach to theirs. We used the multi-class setting from our paper, namely, using MNIST and CIFAR-10 as normal data and other image datasets as anomalous data sets. We also experimented with the simpler one-class setting in ICML18, i.e., using just one class from MNIST (or from CIFAR-10) as normal data. In all cases, ICML 18 perform worse than our approach. Even in the one-class setting, ICML18's accuracy is much worse than our approach in the harder multi-class setting. To confirm that we are correctly using their code, we also ran experiments using exactly their setting with other CIFAR-10 classes as outliers. The results matched their reported results. Note that ICML18 reports AUC, while the minimum value of AUC is 0.5. Therefore their AUC is much higher than their accuracy value on outliers. The results on the simplistic MNIST model are not reported here due to space constraints. 

### The main changes in the code are in src/baseline.py, src/datasets/mnist.py, src/datasets/cifar10.py, src/datasets/preprocessing.py

### We will merge the code with ours once the paper is accepted. 

### The followings are from the README of Deep One-Class Classification

# Deep One-Class Classification
This repository provides the implementation of the *soft-boundary Deep SVDD* and *one-class Deep SVDD* method we used
to perform the experiments for our ”Deep One-Class Classification” ICML 2018 paper. The implementation uses the  
[Theano](http://deeplearning.net/software/theano/) and [Lasagne](https://lasagne.readthedocs.io/en/latest/) libraries.
We plan to publish a [tensorflow](https://www.tensorflow.org) implementation in the future.

## Citation and Contact
You find the PDF of the Deep One-Class Classification ICML 2018 paper at 
[http://proceedings.mlr.press/v80/ruff18a.html](http://proceedings.mlr.press/v80/ruff18a.html).

If you use our work, please also cite the ICML 2018 paper:
```
@InProceedings{pmlr-v80-ruff18a,
  title     = {Deep One-Class Classification},
  author    = {Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Deecke, Lucas and Siddiqui, Shoaib A. and Binder, Alexander and M{\"u}ller, Emmanuel and Kloft, Marius},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  pages     = {4390--4399},
  year      = {2018},
  volume    = {80},
}
```

If you would like to get in touch, please contact [contact@lukasruff.com](mailto:contact@lukasruff.com).

## Abstract

> > Despite the great advances made by deep learning in many machine learning problems, there is a relative dearth of 
> > deep learning approaches for anomaly detection. Those approaches which do exist involve networks trained to perform 
> > a task other than anomaly detection, namely generative models or compression, which are in turn adapted for use in 
> > anomaly detection; they are not trained on an anomaly detection based objective. In this paper we introduce a new 
> > anomaly detection method—Deep Support Vector Data Description—, which is trained on an anomaly detection based
> > objective. The adaptation to the deep regime necessitates that our neural network and training procedure satisfy 
> > certain properties, which we demonstrate theoretically. We show the effectiveness of our method on MNIST and
> > CIFAR-10 image benchmark datasets as well as on the detection of adversarial examples of GTSRB stop signs.

## Installation
This code is written in `Python 2.7` and requires the packages listed in `requirements.txt` in the given versions. 
We recommend to set up a virtual environment in which the packages are then installed, e.g. using `virtualenv`:

```
virtualenv -p python2 env
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
pip install requests
pip install -r requirements.txt
```

We install `Lasagne` first as the `0.2.dev1` version is only available from the GitHub Lasagne repository.

To acitvate/deactivate the environment, run
```
source env/bin/activate
source deactivate
``` 

Make sure that `Theano` floats are set to `float32` by default. This can be done by adding
```
[global]
floatX = float32
```
to the `~/.theanorc` configuration file that is located in your home directory (you may have to create the 
`.theanorc`-file if you have not used `Theano` before).


## Repository structure

### `/data`

Contains the data. We implemented support for MNIST and CIFAR-10:

* MNIST ([http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/))
* CIFAR-10 ([https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html))

To run the experiments, the datasets must be downloaded from the original sources in their original formats 
into the `data` folder.

### `/src`
Source directory that contains all Python code and shell scripts to run experiments.

### `/log`
Directory where the results from the experiments are saved.


## To reproduce results
Change your working directory to `src` and make sure that the respective datasets are downloaded into the
`data` directory. The `src` directory has two subfolders `src/experiments` and `src/scripts`. The `scripts` directory 
provides interfaces to run the implemented methods on different datasets with different settings.

The interface for our Deep SVDD method is given by:
```
sh scripts/mnist_svdd.sh ${device} ${xp_dir} ${seed} ${solver} ${lr} ${n_epochs} ${hard_margin} ${block_coordinate} ${pretrain} ${mnist_normal} ${mnist_outlier};
```
For example to run a MNIST experiment with 0 as the normal class (`mnist_normal = 0`) and all other classes considered 
to be anomalous (`mnist_outlier = -1`), execute the following line:
```
sh scripts/mnist_svdd.sh cpu mnist_0vsall 0 adam 0.0001 150 1 0 1 0 -1;
```
This runs a *one-class Deep SVDD* (`hard_margin = 1` and `block_coordinate = 0`) experiment with pre-training routine 
(`pretrain = 1`) where one-class Deep SVDD is trained for `n_epochs = 150` with the Adam optimizer 
(`solver = adam`) and a learning rate of `lr = 0.0001`. The experiment is executed on `device = cpu` and results are 
exported to `log/mnist_0vsall`. For reproducibility, we set `seed = 0`.

You find descriptions of the various script options within the respective Python files that are called by the shell
scripts (e.g. `baseline.py`).
 
The `experiments` directory can be used to hold shell scripts that start multiple experiments at once. For example 
```
sh experiments/mnist_svdd_exp.sh
```
starts all MNIST one-class Deep SVDD experiments (i.e. ten one-class classification setups each for ten seeds). This 
particular script runs the experiments on 10 CPUs in parallel.


## Disclosure
This implementation is based on the repository 
[https://github.com/oval-group/pl-cnn](https://github.com/oval-group/pl-cnn), which is licensed under the MIT license. 
The *pl-cnn* repository is an implementation of the paper 
[Trusting SVM for Piecewise Linear CNNs](https://arxiv.org/abs/1611.02185) by Leonard Berrada, Andrew Zisserman and 
M. Pawan Kumar, which was an initial inspiration for this research project.

