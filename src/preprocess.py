import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from idx2numpy import convert_from_file
import os

def read_data(path, filelist):
    
    """
    Function loads MNIST data downloaded from https://data.deepai.org/mnist.zip in 
    idx format.
    filelist must be in the order X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = list(map(
            lambda i: convert_from_file(
                os.path.join(path, i)), filelist))

    print(f'X_train.shape: {X_train.shape}')
    print(f'y_train.shape: {y_train.shape}')
    print(f'X_test.shape: {X_test.shape}')
    print(f'y_test.shape: {y_test.shape}')

    return X_train, X_test, y_train, y_test

def X_to_1d(X):

    """
    Function to reshape X from shape (n_samples, n_row, n_col) to shape (n_samples, n_row*n_col)
    """
    X = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
    return X

def rescale_x(X):

    """
    Function rescales X values to the range 0.01 - 1
    """

    X = X*0.99/255 + 0.01
    return X

def rescale_y(y_onehot):
    
    """
    Function to re-scale labels to [0.01, 1]
    """

    y_onehot[y_onehot == 0] = 0.01
    y_onehot[y_onehot == 1] = 0.99
    return y_onehot

def onehot_y(y):

    """
    Function turns categorial labels into one-hot encoded array
    """

    lr = np.arange(10)
    y_onehot = []
    
    for i in range(len(y)):
        y_onehot.append((lr == y[i]).astype(np.float))
    y_onehot = np.array(y_onehot)
    return y_onehot

def plot_6_random_samples(filepath, X, y):
    
    """
    Function to plot 6 random examples
    """

    
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    for i in range(2):
        for j in range(3):
            sample_i = np.random.randint(0, len(X))
            axs[i, j].imshow(
                X[sample_i].reshape((28, 28)), cmap = 'Greys')
            axs[i, j].set_title(y[sample_i])
            axs[i, j].get_xaxis().set_visible(False)
            axs[i, j].get_yaxis().set_visible(False)
    plt.savefig(filepath)

