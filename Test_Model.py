import numpy as np
import pandas as pd
from Training_Model import Training_Model

class Test_Model():
    def __init__(self,X_test,Y_test,coeff):
        self.X_test = X_test
        self.Y_test = Y_test
        self.m_value = coeff[1]
        self.c_value = coeff[0]
    def start(self,X=[]):
        # print(np.subtract([i for i in self.Y_test],self.pred_Y))
        if not X:
            while(1):
                value = float(input("Enter the value:"))
                print(((self.m_value * value) + self.c_value))
                stop = input("Do you want to check another value? Y/N: ")
                if stop !="y":
                    break
        else:
            for x in X:
                print(((self.m_value * x) + self.c_value))
                
            