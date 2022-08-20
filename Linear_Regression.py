# Linear regression ML-Model 
# Thejas Devadiga
# Date : 19/8/22


from Test_Model import Test_Model
from  Training_Model import Training_Model
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    dataFrame = pd.read_csv('./LinearRegressionSheet1.csv')
    
    X = np.array(dataFrame['X'])
    y = np.array(dataFrame['Y'])
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.5,random_state=0)
    
    
    # 70% of data to train the Model 30% to test the model
    trained_Model = Training_Model(X_train,Y_train)
    trained_Model.start(0.00001,7000)
    coeff = trained_Model.get_coefficients()
    tested_Model = Test_Model(X_test,Y_test,coeff)
    tested_Model.start([i for i in X_test])
    tested_Model.start()
if __name__=='__main__':
    main()
