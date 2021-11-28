from typing import Tuple
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
np.set_printoptions(threshold=np.inf)

class Model:
    # Modify your model, default is a linear regression model with random weights
    ID_DICT = {"NAME": "Hengyuan Liu", "BU_ID": "U14069299", "BU_EMAIL": "kevin063@bu.edu"}

    def __init__(self):
        self.theta = None
        self.gradient= None
        self.loss= None
        self.testloss= None
        self.mean=None
    def preprocess(self, X: np.array, y: np.array) -> Tuple[np.array, np.array]:
        ###############################################
        ####      add preprocessing code here      ####
        ###############################################
        X_poly=X[:,[52,532,7,118,1,116,120,25,51]]
        X=np.delete(X,[52,532,7,118,1,116,120,25,51],1)
        #These are the top 20 features we found which can explain more percentage of the variance in data
        #So we use them to provide nonlinear features.
        #poly=PolynomialFeatures(2)
        X_polys=np.square(X_poly)
        X=np.append(X,X_poly,axis=1)
        X=np.append(X,X_polys,axis=1)
        return X, y

    def train(self, X_train: np.array, y_train: np.array):
        """
        Train model with training data
        """
        ###############################################
        ####   initialize and train your model     ####
        ###############################################
        #First we split a valiation set.
        self.mean=np.mean(y_train)
        X_val=X_train[28000:,:]
        y_val=y_train[28000:,:]
        X_train=X_train[:,:]
        y_train=y_train[:,:]
        X_train = np.vstack((np.ones((X_train.shape[0],)), X_train.T)).T
        X_val= np.vstack((np.ones((X_val.shape[0],)), X_val.T)).T
        X_origin=X_train
        X_valorigin=X_val
        #Here we keep a copy of the unnonlinearized data for input of predict
        X_train,y_train=self.preprocess(X_train,y_train)
        #Here I begin with a simple gradient decent apporaching#
        n=X_train.shape[1]
        beginlr=0.01
        beginpenality=1.0001
        #Here we begin with a lr, then iterate to find the best one
        #Here we begin with a penality, then iterate to find the best one
        lrperformance=np.empty((100))
        parameterperformance=np.empty((50))
        parametertheta=np.empty((100,n))
        for muti in range(1):
            #This is used for testing best lr, I will drop it in the final model.
            lr=beginlr+0.002*26
            penality=1/(beginpenality**6.55)
            self.theta = np.random.rand(X_train.shape[1], 1)
            for i in range(1000):
                k=np.array(np.subtract(self.predict(X_origin[:,1:]),y_train))
                self.loss = np.dot(k.transpose(),k)/2/X_train.shape[0]
                self.gradient=((np.dot(k.transpose(),X_train))/X_train.shape[0]).transpose()
                self.theta=penality*self.theta-lr*self.gradient
                kval=np.array(np.subtract(self.predict(X_valorigin[:,1:]),y_val))
                self.testloss=np.mean(np.square(kval))
            #print(self.testloss)
            parametertheta[muti]=self.theta.transpose()
        #predictper=np.mean(X_train,axis=0).reshape((n,1))*self.theta
        #print(self.theta.shape)
        #print(np.mean(X_train,axis=0))
        #print((-predictper.transpose()).argsort()[:,:20])
        #print(parameterperformance)
        #bestpara=np.argmax(parameterperformance)#This is the best lr/penality
        #print(bestpara)
        #print(beginlr+0.0001*bestpara)
        #self.theta=parametertheta[bestpara].transpose()
    

    def predict(self, X_val: np.array) -> np.array:
        """
        Predict with model and given feature
        """
        ###############################################
        ####      add model prediction code here   ####
        ###############################################
        X_val = np.vstack((np.ones((X_val.shape[0],)), X_val.T)).T
        X_val,Y=self.preprocess(X_val,None)
        return 0.8*np.dot(X_val, self.theta)+0.2*self.mean
    
