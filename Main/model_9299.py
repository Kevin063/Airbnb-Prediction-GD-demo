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
    def preprocess(self, X: np.array, y: np.array) -> Tuple[np.array, np.array]:
        ###############################################
        ####      add preprocessing code here      ####
        ###############################################
        X_poly=X[:,[52,532,7,118,1,116,120,25,51,211,33,75,2,212,38,35,8,530,55,62]]
        X=np.delete(X,[52,532,7,118,1,116,120,25,51,211,33,75,2,212,38,35,8,530,55,62],1)
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
        X_train=X_train[:28000,:]
        y_train=y_train[:28000,:]
        X_val=X_train[28000:,:]
        X_train = np.vstack((np.ones((X_train.shape[0],)), X_train.T)).T
        X_origin=X_train
        X_train,y_train=self.preprocess(X_train,y_train)
        #Here I begin with a simple gradient decent apporaching#
        n=X_train.shape[1]
        beginlr=0.00000001
        #Here we begin with a lr, then iterate to find the best one
        lrperformance=np.empty((100))
        lrtheta=np.empty((100,n))
        for muti in range(1):
            #This is used for testing best lr, I will drop it in the final model.
            lr=beginlr*(1.1**muti)
            self.theta = np.random.rand(X_train.shape[1], 1)
            for i in range(1000):
                k=np.array(np.subtract(self.predict(X_origin[:,1:]),y_train))
                self.loss = np.dot(k.transpose(),k)/2/X_train.shape[0]
                self.gradient=((np.dot(k.transpose(),X_train))/X_train.shape[0]).transpose()
                self.theta=0.99*self.theta-0.0001*self.gradient
            lrperformance[muti]=self.loss
            lrtheta[muti]=self.theta.transpose()
        predictper=np.mean(X_train,axis=0).reshape((n,1))*self.theta
        print(self.theta.shape)
        #print(np.mean(X_train,axis=0))
        #print((-predictper.transpose()).argsort()[:,:20])
        #print(lrperformance)
        #print(np.argmax(lrperformance))#This is the best lr
    

    def predict(self, X_val: np.array) -> np.array:
        """
        Predict with model and given feature
        """
        ###############################################
        ####      add model prediction code here   ####
        ###############################################
        X_val = np.vstack((np.ones((X_val.shape[0],)), X_val.T)).T
        X_val,Y=self.preprocess(X_val,None)
        return np.dot(X_val, self.theta)
    
