import numpy as np

from sklearn.decomposition import MiniBatchDictionaryLearning, PCA
from sklearn.feature_extraction.image import PatchExtractor
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision



def center_data(X_train, X_val, X_test,
                mode, offset=None):
    """ center images per channel or per pixel
    """

    if offset is None:
        if mode == "per channel":
            n_channels = np.shape(X_train)[1]
            offset = np.mean(X_train, axis=(0, 2, 3)).reshape(1, n_channels, 1, 1)
        elif mode == "per pixel":
            offset = np.mean(X_train, 0)
        else:
            raise ValueError("Specify mode of centering "
                             "(should be 'per channel' or 'per pixel')")

    X_train -= offset
    X_val -= offset
    X_test -= offset


def normalize_data(X_train, X_val, X_test,
                   mode="per channel", scale=None):
    """ normalize images per channel, per pixel or with a fixed value
    """

    if scale is None:
        if mode == "per channel":
            n_channels = np.shape(X_train)[1]
            scale = np.std(X_train, axis=(0, 2, 3)).reshape(1, n_channels, 1, 1)
        elif mode == "per pixel":
            scale = np.std(X_train, 0)
        elif mode == "fixed value":
            scale = 255.
        else:
            raise ValueError("Specify mode of scaling (should be "
                             "'per channel', 'per pixel' or 'fixed value')")

    X_train /= scale
    X_val /= scale
    X_test /= scale


def rescale_to_unit_interval(X_train, X_val, X_test):
    """
    Scaling all data to [0,1] w.r.t. the min and max in the train data is very
    important for networks without bias units. (data close to zero would
    otherwise not be recovered)
    """

    X_train_min = np.min(X_train)
    X_train_max = np.max(X_train)

    X_train -= X_train_min
    X_val -= X_train_min
    X_test -= X_train_min

    X_train /= (X_train_max - X_train_min)
    X_val /= (X_train_max - X_train_min)
    X_test /= (X_train_max - X_train_min)


def global_contrast_normalization(X_train, X_val, X_test, scale="std"):
    """
    Subtract mean across features (pixels) and normalize by scale, which is
    either the standard deviation, l1- or l2-norm across features (pixel).
    That is, normalization for each sample (image) globally across features. 
    """
    
    assert scale in ("std", "l1", "l2")
    
    na = np.newaxis

    X_train_mean = np.mean(X_train, axis=(1, 2, 3),
                           dtype=np.float32)[:, na, na, na]
    X_val_mean = np.mean(X_val, axis=(1, 2, 3),
                         dtype=np.float32)[:, na, na, na]
    X_test_mean = np.mean(X_test, axis=(1, 2, 3),
                          dtype=np.float32)[:, na, na, na]

    X_train -= X_train_mean
    X_val -= X_val_mean
    X_test -= X_test_mean

    if scale == "std":
        X_train_scale = np.std(X_train, axis=(1, 2, 3),
                               dtype=np.float32)[:, na, na, na]
        X_val_scale = np.std(X_val, axis=(1, 2, 3),
                             dtype=np.float32)[:, na, na, na]
        X_test_scale = np.std(X_test, axis=(1, 2, 3),
                              dtype=np.float32)[:, na, na, na]
    if scale == "l1":
        X_train_scale = np.sum(np.absolute(X_train), axis=(1, 2, 3),
                               dtype=np.float32)[:, na, na, na]
        X_val_scale = np.sum(np.absolute(X_val), axis=(1, 2, 3),
                             dtype=np.float32)[:, na, na, na]
        X_test_scale = np.sum(np.absolute(X_test), axis=(1, 2, 3),
                              dtype=np.float32)[:, na, na, na]
    if scale == "l2":
        # equivalent to "std" since mean is subtracted beforehand
        X_train_scale = np.sqrt(np.sum(X_train ** 2, axis=(1, 2, 3),
                                       dtype=np.float32))[:, na, na, na]
        X_val_scale = np.sqrt(np.sum(X_val ** 2, axis=(1, 2, 3),
                                     dtype=np.float32))[:, na, na, na]
        X_test_scale = np.sqrt(np.sum(X_test ** 2, axis=(1, 2, 3),
                                      dtype=np.float32))[:, na, na, na]

    X_train /= X_train_scale
    X_val /= X_val_scale
    X_test /= X_test_scale


def zca_whitening(X_train, X_val, X_test, eps=0.1):
    """
     Apply ZCA whitening. Epsilon parameter eps prevents division by zero.
    """

    # get shape to later reshape data to original format
    shape_train = X_train.shape
    shape_val = X_val.shape
    shape_test = X_test.shape

    if X_train.ndim > 2:
        X_train = X_train.reshape(shape_train[0], np.prod(shape_train[1:]))
        X_val = X_val.reshape(shape_val[0], np.prod(shape_val[1:]))
        X_test = X_test.reshape(shape_test[0], np.prod(shape_test[1:]))

    # center data
    means = np.mean(X_train, axis=0)
    X_train -= means
    X_val -= means
    X_test -= means

    # correlation matrix
    sigma = np.dot(X_train.T, X_train) / shape_train[0]

    # SVD
    U,S,V = np.linalg.svd(sigma)

    # ZCA Whitening matrix
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + eps)), U.T))

    # Whiten
    X_train = np.dot(X_train, ZCAMatrix.T)
    X_val = np.dot(X_val, ZCAMatrix.T)
    X_test = np.dot(X_test, ZCAMatrix.T)

    # reshape to original shape
    X_train = X_train.reshape(shape_train)
    X_val = X_val.reshape(shape_val)
    X_test = X_test.reshape(shape_test)

    return X_train, X_val, X_test


def make_unit_norm(X_train, X_val, X_test, norm="l2"):
    """
    Normalize each image/tensor to length 1 w.r.t. to the selected norm
    """

    assert norm in ("l1", "l2")

    na = np.newaxis

    if norm == "l2":
        X_train_norms = np.sqrt(np.sum(X_train ** 2, axis=(1, 2, 3),
                                       dtype=np.float32))[:, na, na, na]
        X_val_norms = np.sqrt(np.sum(X_val ** 2, axis=(1, 2, 3),
                                     dtype=np.float32))[:, na, na, na]
        X_test_norms = np.sqrt(np.sum(X_test ** 2, axis=(1, 2, 3),
                                      dtype=np.float32))[:, na, na, na]
    if norm == "l1":
        X_train_norms = np.sum(np.absolute(X_train), axis=(1, 2, 3),
                               dtype=np.float32)[:, na, na, na]
        X_val_norms = np.sum(np.absolute(X_val), axis=(1, 2, 3),
                             dtype=np.float32)[:, na, na, na]
        X_test_norms = np.sum(np.absolute(X_test), axis=(1, 2, 3),
                              dtype=np.float32)[:, na, na, na]

    X_train /= X_train_norms
    X_val /= X_val_norms
    X_test /= X_test_norms


def pca(X_train, X_val, X_test, var_retained=0.95):
    """
    PCA such that var_retained of variance is retained (w.r.t. train set)
    """

    print("Applying PCA...")

    # reshape to 2D if input is tensor
    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
        if X_val.size > 0:
            X_val = X_val.reshape(X_val.shape[0], -1)
        if X_test.size > 0:
            X_test = X_test.reshape(X_test.shape[0], -1)

    pca = PCA(n_components=var_retained)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    if X_val.size > 0:
        X_val = pca.transform(X_val)
    if X_test.size > 0:
        X_test = pca.transform(X_test)

    print("PCA pre-processing finished.")

    return X_train, X_val, X_test


def crop_to_square(image):
    """
    crops an image (n_channels, height, width) to have square size
    with center as in original image
    """

    h, w = image[0, ...].shape
    min_len = min(h, w)

    h_start = (h / 2) - (min_len / 2)
    h_end = (h / 2) + (min_len / 2)
    w_start = (w / 2) - (min_len / 2)
    w_end = (w / 2) + (min_len / 2)

    return image[:, h_start:h_end, w_start:w_end]


def downscale(image, pixels=64):
    """
    downscale image (n_channels, height, width) by factor
    """

    img = Image.fromarray(np.rollaxis(image, 0, 3))

    return np.rollaxis(np.array(img.resize(size=(pixels, pixels))), 2)


def gcn(X, scale="std"):
    """
    Subtract mean across features (pixels) and normalize by scale, which is
    either the standard deviation, l1- or l2-norm across features (pixel).
    That is, normalization for each sample (image) globally across features.
    """

    assert scale in ("std", "l1", "l2")

    na = np.newaxis

    X_mean = np.mean(X, axis=(1, 2, 3), dtype=np.float32)[:, na, na, na]
    X -= X_mean

    if scale == "std":
        X_scale = np.std(X, axis=(1, 2, 3), dtype=np.float32)[:, na, na, na]
    if scale == "l1":
        X_scale = np.sum(np.absolute(X), axis=(1, 2, 3),
                         dtype=np.float32)[:, na, na, na]
    if scale == "l2":
        # equivalent to "std" since mean is subtracted beforehand
        X_scale = np.sqrt(np.sum(X ** 2, axis=(1, 2, 3),
                                 dtype=np.float32))[:, na, na, na]

    X /= X_scale


def getMNISTDataset_small():
    normalize = transforms.Normalize(mean=[0.10026693, 0.10026693, 0.10026693], std=[0.27557483, 0.27557483, 0.27557483])
    MNIST_train_dataset = torchvision.datasets.MNIST(
        root = '../data',
        train = True,
        download = True,
        transform = transforms.Compose(
           [transforms.ToTensor(),
            normalize]
        )
    )
    MNIST_test_dataset = torchvision.datasets.MNIST(
        root = '../data',
        train = False,
        download = True,
        transform = transforms.Compose(
            [transforms.ToTensor(),
             normalize]
        )
    )
    MNIST_all_dataset = torch.utils.data.ConcatDataset([MNIST_train_dataset, MNIST_test_dataset])
    MNIST_data_numpy = np.zeros((len(MNIST_all_dataset), 1,28,28)).astype(np.float32)
    for i in range(len(MNIST_all_dataset)):
        currentImage = MNIST_all_dataset[i][0].numpy()
        tempImage = np.reshape(currentImage, [1,28,28]).astype(np.float32)
        MNIST_data_numpy[i,0,:,:] = tempImage
    print("MNIST data loaded... size = " + str(np.shape(MNIST_data_numpy)))
    return MNIST_data_numpy

def getCIFAR10DatasetOneClass(classNum):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    CIFAR10_train_dataset = torchvision.datasets.CIFAR10(
        root = '../data',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]
        )
    )
    CIFAR10_test_dataset = torchvision.datasets.CIFAR10(
        root = '../data',
        train = False,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]
        )
    )
    CIFAR10_all_dataset = torch.utils.data.ConcatDataset([CIFAR10_train_dataset, CIFAR10_test_dataset])
    CIFAR10_data_numpy = np.zeros((len(CIFAR10_all_dataset), 3,32,32)).astype(np.float32)
    CIFAR10_label_numpy = np.zeros(len(CIFAR10_all_dataset)).astype(np.int32)
    for i in range(len(CIFAR10_all_dataset)):
        currentImage = CIFAR10_all_dataset[i][0].numpy()
        currentLabel = CIFAR10_all_dataset[i][1]
        tempImage = np.reshape(currentImage, [3,32,32]).astype(np.float32)
        CIFAR10_data_numpy[i,:,:,:] = tempImage
        CIFAR10_label_numpy[i] = currentLabel
    idx = np.where(CIFAR10_label_numpy == classNum)[0]
    CIFAR10_data_numpy = CIFAR10_data_numpy[idx,:,:,:]
    CIFAR10_label_numpy = CIFAR10_label_numpy[idx]
    print("CIFAR-10 data loaded... size = " + str(np.shape(CIFAR10_data_numpy)))
    return CIFAR10_data_numpy


def getCIFAR10Dataset():
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    CIFAR10_train_dataset = torchvision.datasets.CIFAR10(
        root = '../data',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,]
        )
    )
    CIFAR10_test_dataset = torchvision.datasets.CIFAR10(
        root = '../data',
        train = False,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize]
        )
    )
    CIFAR10_all_dataset = torch.utils.data.ConcatDataset([CIFAR10_train_dataset, CIFAR10_test_dataset])
    CIFAR10_data_numpy = np.zeros((len(CIFAR10_all_dataset), 3,32,32)).astype(np.float32)

    for i in range(len(CIFAR10_all_dataset)):
        currentImage = CIFAR10_all_dataset[i][0].numpy()
        tempImage = np.reshape(currentImage, [3,32,32]).astype(np.float32)
        CIFAR10_data_numpy[i,:,:,:] = tempImage

    print("CIFAR-10 data loaded... size = " + str(np.shape(CIFAR10_data_numpy)))
    return CIFAR10_data_numpy

def getCIFAR100Dataset():
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    CIFAR100_train_dataset = torchvision.datasets.CIFAR100(
        root = '../data',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,]
        )
    )
    CIFAR100_test_dataset = torchvision.datasets.CIFAR100(
        root = '../data',
        train = False,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize]
        )
    )
    CIFAR100_all_dataset = torch.utils.data.ConcatDataset([CIFAR100_train_dataset, CIFAR100_test_dataset])
    CIFAR100_data_numpy = np.zeros((len(CIFAR100_all_dataset), 3,32,32)).astype(np.float32)
    for i in range(len(CIFAR100_all_dataset)):
        currentImage = CIFAR100_all_dataset[i][0].numpy()
        tempImage = np.reshape(currentImage, [3,32,32]).astype(np.float32)
        CIFAR100_data_numpy[i,:,:,:] = tempImage
    print("CIFAR-100 data loaded... size = " + str(np.shape(CIFAR100_data_numpy)))
    return CIFAR100_data_numpy

def getMNISTDataet():
    normalize = transforms.Normalize(mean=[0.10026693, 0.10026693, 0.10026693], std=[0.27557483, 0.27557483, 0.27557483])
    MNIST_train_dataset = torchvision.datasets.MNIST(
        root = '../data',
        train = True,
        download = True,
        transform = transforms.Compose(
           [transforms.RandomCrop(32, padding=4),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize]
        )
    )
    MNIST_test_dataset = torchvision.datasets.MNIST(
        root = '../data',
        train = False,
        download = True,
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize]
        )
    )
    MNIST_all_dataset = torch.utils.data.ConcatDataset([MNIST_train_dataset, MNIST_test_dataset])
    MNIST_data_numpy = np.zeros((len(MNIST_all_dataset), 3,32,32)).astype(np.float32)
    for i in range(len(MNIST_all_dataset)):
        currentImage = MNIST_all_dataset[i][0].numpy()
        tempImage = np.reshape(currentImage, [3,32,32]).astype(np.float32)
        MNIST_data_numpy[i,:,:,:] = tempImage
    print("MNIST data loaded... size = " + str(np.shape(MNIST_data_numpy)))
    return MNIST_data_numpy

def getSVNHDataset():
    normalize = transforms.Normalize(mean=[0.43834132, 0.44446075, 0.47373202], std=[0.19815595, 0.20110732, 0.1970344])
    SVHN_train_dataset = torchvision.datasets.SVHN(
        root = '../data',
        split = 'train',
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,]
        )
    )
    SVHN_test_dataset = torchvision.datasets.SVHN(
        root = '../data',
        split = 'test',
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize]
        )
    )
    SVHN_all_dataset = torch.utils.data.ConcatDataset([SVHN_train_dataset, SVHN_test_dataset])
    SVHN_data_numpy = np.zeros((len(SVHN_all_dataset), 3,32,32)).astype(np.float32)
    for i in range(len(SVHN_all_dataset)):
        currentImage = SVHN_all_dataset[i][0].numpy()
        tempImage = np.reshape(currentImage, [3,32,32]).astype(np.float32)
        SVHN_data_numpy[i,:,:,:] = tempImage
    print("SVHN data loaded... size = " + str(np.shape(SVHN_data_numpy)))
    return SVHN_data_numpy

def getSVHNDataset_small():
    normalize = transforms.Normalize(mean=[0.43834132, 0.44446075, 0.47373202], std=[0.19815595, 0.20110732, 0.1970344])
    SVHN_train_dataset = torchvision.datasets.SVHN(
        root = '../data',
        split = 'train',
        download = True,
        transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize,]
        )
    )
    SVHN_test_dataset = torchvision.datasets.SVHN(
        root = '../data',
        split = 'test',
        download = True,
        transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize]
        )
    )
    SVHN_all_dataset = torch.utils.data.ConcatDataset([SVHN_train_dataset, SVHN_test_dataset])
    SVHN_data_numpy = np.zeros((len(SVHN_all_dataset), 1,28,28)).astype(np.float32)
    for i in range(len(SVHN_all_dataset)):
        currentImage = SVHN_all_dataset[i][0].numpy()
        tempImage = np.reshape(currentImage, [1,28,28]).astype(np.float32)
        SVHN_data_numpy[i,0,:,:] = tempImage
    print("SVHN data loaded... size = " + str(np.shape(SVHN_data_numpy)))
    return SVHN_data_numpy

def getCIFAR100Dataset_small():
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    CIFAR100_train_dataset = torchvision.datasets.CIFAR100(
        root = '../data',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize,]
        )
    )
    CIFAR100_test_dataset = torchvision.datasets.CIFAR100(
        root = '../data',
        train = False,
        download = True,
        transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize]
        )
    )
    CIFAR100_all_dataset = torch.utils.data.ConcatDataset([CIFAR100_train_dataset, CIFAR100_test_dataset])
    CIFAR100_data_numpy = np.zeros((len(CIFAR100_all_dataset), 1,28,28)).astype(np.float32)
    for i in range(len(CIFAR100_all_dataset)):
        currentImage = CIFAR100_all_dataset[i][0].numpy()
        tempImage = np.reshape(currentImage, [1,28,28]).astype(np.float32)
        CIFAR100_data_numpy[i,0,:,:] = tempImage
    print("CIFAR-100 data loaded... size = " + str(np.shape(CIFAR100_data_numpy)))
    return CIFAR100_data_numpy

def getCIFAR10Dataset_small():
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    CIFAR10_train_dataset = torchvision.datasets.CIFAR10(
        root = '../data',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize,]
        )
    )
    CIFAR10_test_dataset = torchvision.datasets.CIFAR10(
        root = '../data',
        train = False,
        download = True,
        transform = transforms.Compose([
            transforms.CenterCrop(28),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize]
        )
    )
    CIFAR10_all_dataset = torch.utils.data.ConcatDataset([CIFAR10_train_dataset, CIFAR10_test_dataset])
    CIFAR10_data_numpy = np.zeros((len(CIFAR10_all_dataset), 1,28,28)).astype(np.float32)
    for i in range(len(CIFAR10_all_dataset)):
        currentImage = CIFAR10_all_dataset[i][0].numpy()
        tempImage = np.reshape(currentImage, [1,28,28]).astype(np.float32)
        CIFAR10_data_numpy[i,0,:,:] = tempImage
    print("CIFAR-10 data loaded... size = " + str(np.shape(CIFAR10_data_numpy)))
    return CIFAR10_data_numpy

def get_norm_for_cifar10():
    # X_normal = getCIFAR10Dataset()
    X_normal = getCIFAR10DatasetOneClass(0)
    perm_norm = np.random.permutation(np.shape(X_normal)[0])
    X_normal = X_normal[perm_norm]
    X_normal = X_normal[0:6000,:,:,:]
    y_normal = np.zeros(np.shape(X_normal)[0], dtype=np.uint8)
    return X_normal, y_normal

def get_outlier_for_cifar10():
    # X_normal = getCIFAR10Dataset()
    X_normal = getCIFAR10DatasetOneClass(0)
    perm_norm = np.random.permutation(np.shape(X_normal)[0])
    X_normal = X_normal[perm_norm]
    X_normal = X_normal[0:500,:,:,:]
    X_outlier = getSVNHDataset()
    perm_norm = np.random.permutation(np.shape(X_outlier)[0])
    X_outlier = X_outlier[perm_norm]
    X_outlier = X_outlier[0:4500,:,:,:]
    y_normal = np.zeros(np.shape(X_normal)[0], dtype=np.uint8)
    y_outlier = np.ones(np.shape(X_outlier)[0], dtype=np.uint8)
    return X_normal,X_outlier, y_normal, y_outlier

def get_norm_for_mnist():
    X_normal = getMNISTDataset_small()
    perm_norm = np.random.permutation(np.shape(X_normal)[0])
    X_normal = X_normal[perm_norm]
    X_normal = X_normal[0:10000,:,:,:]
    y_normal = np.zeros(np.shape(X_normal)[0], dtype=np.uint8)
    return X_normal, y_normal

def get_outlier_for_mnist():
    X_normal = getMNISTDataset_small()
    perm_norm = np.random.permutation(np.shape(X_normal)[0])
    X_normal = X_normal[perm_norm]
    X_normal = X_normal[0:10000,:,:,:]
    X_outlier = getSVHNDataset_small()
    perm_norm = np.random.permutation(np.shape(X_outlier)[0])
    X_outlier = X_outlier[perm_norm]
    X_outlier = X_outlier[0:10000,:,:,:]
    y_normal = np.zeros(np.shape(X_normal)[0], dtype=np.uint8)
    y_outlier = np.ones(np.shape(X_outlier)[0], dtype=np.uint8)
    return X_normal,X_outlier, y_normal, y_outlier

def extract_norm_and_out(X, y, normal, outlier):
    '''
    
    :param X: numpy array with data features 
    :param y: numpy array with labels
    :param normal: list with labels declared normal
    :param outlier: list with labels declared outliers
    :return: X_normal, X_outlier, y_normal, y_outlier
    '''
    # getMNISTDataset_small()
    print("normal: " + str(normal))
    print("outlier: " + str(outlier))
    idx_normal = np.any(y[..., None] == np.array(normal)[None, ...], axis=1)
    idx_outlier = np.any(y[..., None] == np.array(outlier)[None, ...], axis=1)

    X_normal = X[idx_normal]
    # print(np.shape(X_normal))
    y_normal = np.zeros(np.sum(idx_normal), dtype=np.uint8)
    # print(np.shape(y_normal))
    X_outlier = X[idx_outlier]
    y_outlier = np.ones(np.sum(idx_outlier), dtype=np.uint8)
    # print(np.shape(X_outlier))
    # print(np.shape(y_outlier))
    return X_normal, X_outlier, y_normal, y_outlier


def learn_dictionary(X, n_filters, filter_size, n_sample=1000,
                     n_sample_patches=0, **kwargs):
    """
    learn a dictionary of n_filters atoms from n_sample images from X
    """

    n_channels = X.shape[1]

    # subsample n_sample images randomly
    rand_idx = np.random.choice(len(X), n_sample, replace=False)

    # extract patches
    patch_size = (filter_size, filter_size)
    patches = PatchExtractor(patch_size).transform(
        X[rand_idx, ...].reshape(n_sample, X.shape[2], X.shape[3], X.shape[1]))
    patches = patches.reshape(patches.shape[0], -1)
    patches -= np.mean(patches, axis=0)
    patches /= np.std(patches, axis=0)

    if n_sample_patches > 0 and (n_sample_patches < len(patches)):
        np.random.shuffle(patches)
        patches = patches[:n_sample_patches, ...]

    # learn dictionary
    print('Learning dictionary for weight initialization...')

    dico = MiniBatchDictionaryLearning(n_components=n_filters, alpha=1, n_iter=1000, batch_size=10, shuffle=True,
                                       verbose=True, **kwargs)
    W = dico.fit(patches).components_
    W = W.reshape(n_filters, n_channels, filter_size, filter_size)

    print('Dictionary learned.')

    return W.astype(np.float32)
