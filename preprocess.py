import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

# separate labels and input in both sets
def splitTrainTest(traindata, testdata):
    traindata = np.loadtxt(traindata, delimiter=',')
    testdata = np.loadtxt(testdata, delimiter=',')
    ytrain = traindata[:, 0]
    ytest = testdata[:, 0]
    Xtrain = traindata[:, 1:]
    Xtest = testdata[:, 1:]
    return ytrain, ytest, Xtrain, Xtest

# re-scale pixel values to [0.01, 1]
def rescaleX(Xtrain, Xtest):
    Xtrain = Xtrain*0.99/255 + 0.01
    Xtest = Xtest*0.99/255+ 0.01
    return Xtrain, Xtest

# one-hot encoding of labels
def onehoty(ytrain, ytest):    
    lr = np.arange(10)
    ytrain_onehot = []
    ytest_onehot = []
    for i in range(len(ytrain)):
        ytrain_onehot.append((lr == ytrain[i]).astype(np.float))
    for i in range(len(ytest)):
        ytest_onehot.append((lr == ytest[i]).astype(np.float))
    ytrain_onehot = np.array(ytrain_onehot)
    ytest_onehot = np.array(ytest_onehot)
    return ytrain_onehot, ytest_onehot

# re-scale labels to [0.01, 1]
def rescaley(ytrain_onehot, ytest_onehot):
    ytrain_onehot[ytrain_onehot == 0] = 0.01
    ytrain_onehot[ytrain_onehot == 1] = 0.99
    ytest_onehot[ytest_onehot == 0] = 0.01
    ytest_onehot[ytest_onehot == 1] = 0.99
    return ytrain_onehot, ytest_onehot
