# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 01:20:51 2018

@author: sony
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

lists= ["length","diameter","height","whole","shucked","viscera","shell","rings"]

a= pd.read_csv(r"C:\Users\sony\Downloads\Abalone\abalone.data",index_col=0, sep=",",names= lists)
b = pd.read_csv(r"C:\Users\sony\Downloads\Abalone\abalone.domain")


X=a.iloc[:,: a.shape[1]-1]
Y=a.rings
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.33,random_state=42,shuffle=False)

linreg=LinearRegression()
linreg.fit(x_train,y_train)
result=linreg.predict(x_test)

print(np.sqrt(mean_squared_error(y_test,result)))




