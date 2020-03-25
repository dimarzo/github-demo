# my first python

# DATAFRAMES

import numpy as np

import pandas as pd

A=np.random.randn(5,3)

ds=pd.DataFrame(A)

ds

ds.index

ds.columns

ds.shape

ds.head(2)

ds.columns=["col1","col2","col3"]

ds

ds.describe()

ds.sort_values('col1')

'''
PYTHON STARTS COUNTING FROM 0

'''

# DATA SELECTION


ds[0:1]

ds.loc[0:2,['col1','col2']]

ds.iloc[0:2,0:2]

ds.iat[0,0]

#DATA FILTERING

ds[ds.col1>0]



#MATH FUNCTIONS

import math

math.fsum([3,4,5])

math.sqrt(112)

import numpy

a=np.random.randn(100,1) 

numpy.std(a)

numpy.mean(a)

numpy.median(a)



## DATA MANIPULATION


a=['1','3000','8']

[int(v) for v in a]

[map(int,a)] ##?????

int('4')

e=str(6)

math.isnan(4)  ##???????

g='ciao'

len(g)

######## MAP

def calc(n): return n*n

numbers = (1, 2, 3, 4)

map(calc, numbers) #????

[calc(x) for x in numbers]

#########

### DATA VISUALITATION

from sklearn import datasets 

iris = datasets.load_iris()

import matplotlib.pyplot as plt

iris = iris.data

plt.scatter(iris[:,1],iris[:,2])

plt.boxplot(iris[:,1])

plt.hist(iris[:,1])

x = iris[:,1] 

y = iris[:,2]

size = iris[:,3]

plt.scatter(x,y,s=size * 200)




 





