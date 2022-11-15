
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tslearn.metrics import dtw
from tslearn.metrics import soft_dtw
from tslearn.clustering import TimeSeriesKMeans

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

import sys


def optimal_k(dataframe,min,max):
  Sum_of_squared_distances = []
  K = range(min,max)
  for k in K:
      km = KMeans(n_clusters=k)
      km = km.fit(dataframe)
      Sum_of_squared_distances.append(km.inertia_)

  plt.plot(K, Sum_of_squared_distances, 'bx-')
  plt.xlabel('k')
  plt.ylabel('Sum_of_squared_distances')
  plt.title('Elbow Method For Optimal k')
  plt.show()
  


def kmeans(csv,k,k_seeds,xaxis, yaxis,x_scaler = None, y_scaler= None):
  
  # preprossing 
  df = pd.read_csv(f'{csv}',lineterminator='\n')

  df[f'{xaxis}'] = pd.to_datetime(df[f'{xaxis}'])
  # df =  df[df[f'{yaxis}'] != 0]
  df['ts'] = df[f'{xaxis}'].values.astype(np.int64) // 10 ** 9

  # scaling options: MinMaxScaler,minmax_scale,MaxAbsScaler,StandardScaler,RobustScaler,Normalizer,QuantileTransformer,PowerTransformer
  if x_scaler is not None:
    df[[f'ts']] = x_scaler().fit_transform(df[[f'ts']])
  if y_scaler is not None:
    df[[f'{yaxis}']] = y_scaler().fit_transform(df[[f'{yaxis}']])

  data = df[['ts',f'{yaxis}']]


  
  model = KMeans(init="random",n_clusters=k,n_init=10,max_iter=300,random_state=4)
  model.fit(data)

  y=model.predict(data)

  centroids = model.cluster_centers_

  #plotting the data
  plt.style.use('seaborn')
  plt.rcParams['font.size'] = 14 
  plt.rcParams['axes.labelsize'] = 12 
  plt.rcParams['axes.labelweight'] = 'bold' 
  plt.rcParams['image.cmap'] = 'jet' 
  plt.rcParams['image.interpolation'] = 'none' 
  plt.rcParams['figure.figsize'] = (12, 10) 
  plt.rcParams['axes.grid']=True
  data_c = data.copy()
  data_c['Class']=y
  print(data_c)

  sns.scatterplot(x='ts',y=f'{yaxis}',hue='Class',data=data_c,palette='plasma')
  plt.scatter(centroids[:,0] , centroids[:,1] , s = 250, color = 'k',alpha = 0.9)



  return plt.show()



if __name__ == "__main__":
  # example input: kmeans("Month_Value_1.csv",2,3,"Day","Sales_quantity",StandardScaler,StandardScaler)
  # input function parameters kmeans(str,str,int,int,str,str,no quotes or None, no quotes or None)
  graph = kmeans(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7])
