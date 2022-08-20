import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
 
class Training_Model():
    def __init__(self,X_train,Y_train):
        
        self.train_X = X_train
        self.train_Y = Y_train
        
        
    def start(self,rate_of_learning,steps):
        self.regressor = Linear_Regression(self.train_X, self.train_Y)
        print(self.train_X.shape,self.train_Y.shape)
        print("Train model...")
        iterations = 0     
        costs = []
        self.predicted_Y = self.regressor.predictor()
        
        
        self.regressor.best_fit_plot(self.predicted_Y,"INITIAL FIT")
        while 1:
            self.predicted_Y = self.regressor.predictor()
            cost = self.regressor.cost_computed(self.predicted_Y)
            costs.append(cost)
            self.regressor.coefficient_update(rate_of_learning)
            
            iterations+=1
            if iterations % steps == 0:
                print("Iterations elapsed : ", iterations)
                print("Accuracy : ", self.regressor.get_Accuracy(self.predicted_Y))
                stop = input("Do you want to stop trainingModel (y/*)??") 
                if stop =="y":
                    break
        
        self.regressor.best_fit_plot(self.predicted_Y, "FINAL FIT")
        self.regressor.print_coefficients()
        print("The model is trained successfully!!!")
        plt.plot(np.array(range(iterations)), costs,color='b')
        plt.show()
        print("Use value_Predictor method to predict the values...")
    def value_Predictor(self,values):
        return self.regressor.predictor(values)
    def get_coefficients(self):
        return self.regressor.get_coefficients()
        


        
class Linear_Regression():
    def __init__(self, train_X, train_Y):
        self.X = train_X
        self.Y = train_Y
        self.B0 = 0
        self.B1 = 0
        
        
    def best_fit_plot(self,predicted_Y,label):
        plt.figure(label)
        plt.scatter(self.X,self.Y,color='red')
        plt.plot(self.X,predicted_Y,color='green')
        plt.show()
        
    def predictor(self, X=[]):
        predicted_Y = np.array([])
        
        if not X: 
            X=self.X
        b0 = self.B0
        b1 = self.B1
        for x in X:
            predicted_Y = np.append(predicted_Y,b0 + ( b1 * x ) )
        return predicted_Y
    
    def get_Accuracy(self,predicted_Y):
    
        predict,err = predicted_Y, self.Y
        n = len(predicted_Y)
        return 1-sum([abs(predict[i]-err[i])/err[i]
                      for i in range(n) 
                      if err[i] !=0]
                    )/n
    
    def coefficient_update(self,rate_of_learning):
        predicted_Y = self.predictor()
        Y = self.Y
        m = len(Y)
        self.B0 = self.B0 -(rate_of_learning * ((1/m) * np.sum((predicted_Y - Y) )))        
        self.B1 = self.B1 -(rate_of_learning * ((1/m) * np.sum((predicted_Y - Y) * self.X)))        
        
    def cost_computed(self,predicted_Y):
        m = len(self.Y)
        cost = (1/2*m) *((np.sum(predicted_Y - self.Y) **2))
        return cost 
    def print_coefficients(self):
        print("The coefficients are:",self.B0,self.B1)
    def get_coefficients(self):
        return [self.B0,self.B1]
    
    
    
        
    