__author__ = 'raul'

from math import log #Se utiliza para calcular los rendimientos logaritmicos
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

#Se obtienen las ventanas bursatiles
def windows(df):

    start_date=df.iloc[0][1]
    end_date=df.iloc[0][df.shape[1]]
    #Se obtienen el numero de ventanas bursatiles
    delta_year=end_date.year-start_date.year
    if delta_year > 1:
        num_ventanas= (13-start_date.month) + (delta_year-1)*(12) + end_date.month
    else:
        num_ventanas=end_date.month-start_date.month

    #Se obtienen las ventanas bursatiles
    ventana=[]

    j=0
    for i in xrange(2,df.shape[1]):
        if df.iloc[0][i].month>df.iloc[0][i-1].month or (df.iloc[0][i].month==1 and df.iloc[0][i-1].month==12):
            ventana.append(df.iloc[:len(df),j:i])
            j=i

    if len(ventana)<num_ventanas:
        ventana.append(df.iloc[:len(df),j:df.shape[1]])


    #Se elimina la fecha para poder hacer el clustering con KMeans
    for i in xrange(len(ventana)):
        ventana[i]=ventana[i].drop('Fecha')

    return ventana

def cluster_data(data, n_clusters=8):
    cluster_model = KMeans(n_clusters)
    prediction = cluster_model.fit_predict(data)
    return prediction, cluster_model

def measure_error(prediction, model, c_data):
    error_score = []
    for counter in range(len(c_data)):
        true_val = c_data.drop("Cluster",1).values[counter]
        center_val = model.cluster_centers_[c_data["Cluster"][counter]]

        error_score.append(np.mean(np.abs(true_val - center_val)) / np.mean(center_val))

    cluster_counts = c_data["Cluster"].value_counts()

    return np.average(error_score), len(cluster_counts[cluster_counts==1])

def clustering(ventana, n_clusters=8):
    error=[]
    failed_clusters=[]
    for i in xrange(len(ventana)):
        appending_cluster, appending_model = cluster_data(ventana[i],n_clusters)
        ventana[i]['Cluster']=appending_cluster #Agrego la columna Cluster a la ventana[i]

        #Se mueve la ultima columna (Cluster) al principio
        cols = ventana[i].columns.tolist()
        cols = cols[-1:] + cols[:-1]
        ventana[i]=ventana[i][cols]

        appending_error, appending_failed_clusters= measure_error(appending_cluster, appending_model, ventana[i])
        error.append(appending_error)
        failed_clusters.append(appending_failed_clusters)
    error_df=pd.DataFrame(error)
    error_df=error_df.rename(columns={0:'Error'})
    error_df['Failed Cluster']=failed_clusters
    return ventana, error_df

precios=pd.read_excel('OPIIFPrecios.xlsx')
precios=precios.convert_objects(convert_numeric=True)
precios['Fecha']=pd.to_datetime(precios['Fecha'])
precios=precios.T

df=precios.copy()
for i in xrange(1,len(precios)):
    for j in xrange(1,precios.shape[1]):
        df.iloc[i][j]=log(precios.iloc[i][j]/precios.iloc[i][j-1])
df=df.drop(0,axis=1)

ventana=windows(df)

n_clusters=5
ventana, error_df = clustering(ventana,n_clusters)