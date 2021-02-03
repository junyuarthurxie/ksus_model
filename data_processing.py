import torch
import numpy as np
import os, os.path
import mnist_reader

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_data(train_size, test_size):
    total_size = train_size + test_size
    if torch.cuda.is_available():
        #data, _ = mnist_reader.load_mnist('/content/', kind='train')
        data, _ = mnist_reader.load_mnist(dir_path + '/data/fashion', kind='train')
    else:
        data, _ = mnist_reader.load_mnist(dir_path + '/data/fashion', kind='train')
    data_demo = data[data.shape[0] - 1, :]
    data_demo = np.resize(data_demo,  (28, 28))
    data_demo = (np.pad(data_demo, ((2, 2), (2, 2)), 'constant') / 255).astype('float64') 

    data = data[0 : total_size, :]
    data = np.resize(data, (total_size, 28, 28))
    data = (np.pad(data, ((0, 0), (2, 2), (2, 2)), 'constant') / 255).astype('float64')   # get to 32x32
    

    data_train = data[0 : train_size]
    data_test = data[train_size : total_size]
    
    return (data_train, data_test, data_demo)  
    #dimension (n, h, w) (n, h, w) (h, w)

