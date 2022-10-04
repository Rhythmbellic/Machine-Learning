from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
a=float(input("Enter the lenth of sepal in cm\n"))
b=float(input("Enter the width of sepal in cm\n"))
c=float(input("Enter the lenth of petal in cm\n"))
d=float(input("Enter the width of petal in cm\n"))
iris=datasets.load_iris()
features=iris.data
labels=iris.target
clf=KNeighborsClassifier()
clf.fit(features,labels)
pred=clf.predict([[a,b,c,d]])
if (pred==0):
    print("This iris is setosa\n")
elif (pred==1):
    print("This iris is versicolour\n")
else:
    print("This iris is virginica\n")