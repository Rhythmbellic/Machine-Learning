import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
diabetes=datasets.load_diabetes()
#print(diabetes.keys())
#(['data', 'target', 'frame', 'DESCR', 'feature_names', '
# data_filename', 'target_filename', 'data_module']) 
x=diabetes.data
x_train=x[:-30]
x_test=x[-30:]
y=diabetes.target
y_train=y[:-30]
y_test=y[-30:]
model=linear_model.LinearRegression()
model.fit(x_train,y_train)
diabete_y_predict=model.predict(x_test)
print(f"Mean square error is:{mean_squared_error(y_test,diabete_y_predict)}")
print(f"weights:{model.coef_}")
print(f"Intercept:{model.intercept_}")
