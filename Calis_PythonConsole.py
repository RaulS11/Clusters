__author__ = 'raul'
def calis_func(a):
    return a+5

a=5
b=calis_func(a)
c,d=20

import pandas as pd

df=pd.DataFrame({'x':[1,2,3],'y':['x','x','y'],'z':[10,20,30]})
df1=df.pivot(index='x',columns='y',values='z')

from sklearn.preprocessing import normalize
df2=pd.DataFrame({'x':[1,2,3],'y':[5,10,15],'z':[10,20,30]})
norm_df=normalize(df2.values.astype(float),norm='l2',copy=True)

from sklearn.cluster import KMeans
cluster=KMeans(8)

if a<c: c=15

mean=[[1],[2],[3],[4],[5]]

import numpy as np
mean2=np.array([1,2,3,4,5])
t_mean2=mean2.transpose()
t_mean2.shape=(5,1)