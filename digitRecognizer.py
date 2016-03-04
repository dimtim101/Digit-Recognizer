import numpy as np
import math
import idx2numpy
import random
import operator
import pickle
import idxutil
from pandas import DataFrame


def sig(t):
    if t > 100:
        return 1

    if t < -100:
        return 0

    return 1/(1 + math.exp(-t))

sig = np.vectorize(sig, otypes=[np.float])

class NeuralNetwork:
   
    def __init__(self,nI,nH,nO):
        self.nI = nI
        self.nH = nH
        self.nO = nO
        self.eta = 0.5

        self.V = (np.random.rand(nH,nI) * 2 - 1) * 1.0/(nI ** 0.5) # weights hidden-input
        self.W = (np.random.rand(nO,nH) * 2 - 1) * 1.0/(nH ** 0.5) # weights output-hidden
        self.b1 = (np.random.rand(nH) *2 -1) * 1.0/(nI ** 0.5)  # bias weights hidden-input
        self.b2 = (np.random.rand(nO) *2 -1) * 1.0/(nH ** 0.5)  # bias weights output-hidden


        self.x = np.array([0.0] * nI)   # Input
        self.z = np.array([0.0] * nO)   # Hidden activation output
        self.t = np.array([0.0] * nO)   # Output
    
   
    def feed_forward(self):
        self.y = sig(np.dot(self.V,self.x) + self.b1)
        self.z = sig(np.dot(self.W,self.y) + self.b2)

    def back_prop(self):

        self.delk = (self.t-self.z) * (1-self.z) * self.z
        self.delj = np.dot(self.W.T,self.delk) * (1-self.y) * self.y

        self.W += self.eta * np.outer(self.delk,self.y)
        self.b2 += self.eta * self.delk
        self.V +=  self.eta * np.outer(self.delj,self.x)
        self.b1 += self.eta * self.delj

    def train(self,feature,label):
        self.x = feature
        self.t = label
        self.feed_forward()
        self.back_prop()

        e = self.t - self.z
        return 0.5 * np.dot(e,e)

    def get_output(self,test_feature):
        y = sig(np.dot(self.V,test_feature) + self.b1)
        z = sig(np.dot(self.W,y) + self.b2)
        return z 

    def info(self):
        print 'Two - Layer Neural Network'
        print 'No. of inputs',self.nI
        print 'No. of hidden neurons',self.nH
        print 'No. of ouput',self.nO



def test(nn):
    (images,labels) = idxutil.load_mnsit('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte')
    
    conmat = np.array([[0] * 10]*10)
    correct = 0
    for image,actual in zip(images,labels):
        image = image/255.0
        z = nn.get_output(image)
        predicted, value = max(enumerate(z), key=operator.itemgetter(1))

        if actual == predicted:
            correct+=1
        conmat[actual][predicted]  +=1

    total = len(images)
    print '\nOverall Accuracy',float(correct)/total
    print '\nConfusion Matrix'
    print DataFrame(conmat)
    for label in range(10):
        TP = conmat[label][label] 
        FP = conmat[:,label].sum() - TP
        FN = conmat[label].sum() - TP
        TN = conmat.sum() - TP - FN - FP

        print '\nClass:', label
        print 'TP',TP
        print 'FP',FP
        print 'FN',FN
        print 'TN',TN
        print 'Accuracy:',float(TP+TN)/total
        print 'Misclassification rate:',float(FP+FN)/total
        print 'Sensitivity/Recall:', float(TP)/(TP+FN)
        print 'Specificity', float(TN)/(TN+FP)
        print 'Pecision',  float(TP)/(TP+FP)


def digitRecogniser():

    nn = NeuralNetwork(28 * 28,100,10)
    totalEpochs = 5
    nn.info()
    (images,label) = idxutil.load_mnsit('train-images-idx3-ubyte','train-labels-idx1-ubyte')
    print '\nEpoch     Squared Error'
    for epoch in range(totalEpochs):
        err = 0.0
        for image,o in zip(images,label):
            t = np.array([0.15] * 10)   # Output
            t[o] = 0.95
            image = image/255.0
            err += nn.train(image,t)
        print '%-5d     %-10f' % (epoch+1,err)
            

    test(nn)

digitRecogniser()


'''
X=2-input XOR
for i in range(1000):
    nn.train([0,0],[1,0])
    nn.train([0,1],[0,1])
    nn.train([1,0],[0,1])
    nn.train([1,1],[1,0])


print nn.get_output([0,0])
print nn.get_output([0,1])
print nn.get_output([1,0])
print nn.get_output([1,1])
'''
