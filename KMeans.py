__author__ = 'raul'

from math import log #Se utiliza para calcular los rendimientos logaritmicos
from itertools import repeat
from collections import Counter #Para encontrar la moda
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
    cluster_model = KMeans(n_clusters, n_init=20, max_iter=500)
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

def clustering(ventana, n_clusters=3):
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

def clustering_fallidos(ventana, error_df, indices_fallidos, n_clusters=3):
    for i in indices_fallidos:
        ventana[i]=ventana[i].drop('Cluster',axis=1)
        appending_cluster, appending_model = cluster_data(ventana[i],n_clusters)
        ventana[i]['Cluster']=appending_cluster #Agrego la columna Cluster a la ventana[i]

        appending_error, appending_failed_clusters= measure_error(appending_cluster, appending_model, ventana[i])
        if appending_error < error_df.iloc[i][0] and appending_failed_clusters < error_df.iloc[i][1]:
            error_df.iloc[i][0]=appending_error
            error_df.iloc[i][1]=appending_failed_clusters
    return ventana, error_df

def window_clusters(clusters, j, n_clusters=3 ):
    cluster_ventana=[ [] for i in repeat(None,n_clusters) ]
    for i in xrange(len(clusters)):
        for k in xrange(n_clusters):
            if clusters.iloc[i][j]==k:
                cluster_ventana[k].append(i)
    return cluster_ventana

def frecuencia(clusters_ventana, n_clusters=3):
    a=np.empty([1,n_clusters], dtype=set) #Clusters de la ventana 0.
    a=a[0]
    b=np.empty([n_clusters, len(clusters_ventana)-1], dtype=set) #No se toma en cuenta la ventana 0 (esa esta en a).
    c=np.empty([n_clusters,len(clusters_ventana)-1,n_clusters], dtype=set) #No se toma en cuenta la ventana 0.
    f=np.empty([n_clusters,len(clusters_ventana)-1,n_clusters]) #Matriz de frecuencias.
    for k in xrange(n_clusters):
        a[k]=set(clusters_ventana[0][k])
    for j in xrange(1,len(clusters_ventana)):
        for l in xrange(n_clusters):
            b[l,j-1]=set(clusters_ventana[j][l])
    for k in xrange(n_clusters):
        for j in xrange(len(clusters_ventana)-1):
            for l in xrange(n_clusters):
                c[k,j,l]=a[k]&b[l,j]
    for k in xrange(n_clusters):
        for j in xrange(len(clusters_ventana)-1):
            for l in xrange(n_clusters):
                f[k,j,l]=len(c[k,j,l])

    freq_indexes=[ [] for i in repeat(None,n_clusters)]
    for k in xrange(n_clusters):
        for j in xrange(len(clusters_ventana)-1):
            thing=np.argwhere(f[k,j]==np.amax(f[k,j]))
            another_thing=list()
            for equis in xrange(len(thing)):
                another_thing.append(thing[equis][0])
            freq_indexes[k].append(another_thing)
    return freq_indexes

def criss_cross(matriz_freq, matriz_len, c):
    matriz_freq_original=matriz_freq.copy()
    for j in xrange(matriz_freq.shape[1]):
        for k in xrange(matriz_freq.shape[0]):
            if k != c:
                if matriz_len[c,j]>matriz_len[k,j]:
                    subset=matriz_freq_original[c,j]&matriz_freq_original[k,j]
                    matriz_freq[c,j]=matriz_freq[c,j]-subset
                elif matriz_len[c,j]<matriz_len[k,j]:
                    subset=matriz_freq_original[c,j]&matriz_freq_original[k,j]
                    matriz_freq[k,j]=matriz_freq[k,j]-subset
                elif matriz_len[c,j]==matriz_len[k,j]:
                    subset=matriz_freq_original[c,j]&matriz_freq_original[k,j]
                    matriz_freq[c,j]=matriz_freq[c,j]-subset
                    matriz_freq[k,j]=matriz_freq[k,j]-subset
    return matriz_freq

precios=pd.read_excel('OPIIFPrecios2.xlsx')
precios=precios.convert_objects(convert_numeric=True)
precios['Fecha']=pd.to_datetime(precios['Fecha'])
precios=precios.T

df=precios.copy()
for i in xrange(1,len(precios)):
    for j in xrange(1,precios.shape[1]):
        df.iloc[i][j]=log(precios.iloc[i][j]/precios.iloc[i][j-1])
df=df.drop(0,axis=1)

ventana=windows(df)

n_clusters=3
ventana, error_df = clustering(ventana,n_clusters)

max_iteraciones=1000
iteracion=0
while iteracion < max_iteraciones:
    indices_fallidos=[]
    for i in xrange(len(error_df)):
        if error_df.iloc[i][0] > 20 or error_df.iloc[i][0] < -20 or error_df.iloc[i][1] > 0:
            indices_fallidos.append(i)
    iteracion=iteracion+1
    if indices_fallidos==[]:
        iteracion=max_iteraciones
    ventana, error_df=clustering_fallidos(ventana,error_df, indices_fallidos,n_clusters)

frames=list()
for i in xrange(len(ventana)):
    frames.append(ventana[i]['Cluster'])
clusters=pd.concat(frames,axis=1)

clusters_ventana=list()
for j in xrange(clusters.shape[1]):
    clusters_ventana.append(window_clusters(clusters,j,n_clusters))

freq=frecuencia(clusters_ventana,n_clusters)

matriz_freq=np.empty([n_clusters,len(clusters_ventana)-1], dtype=set)
matriz_len=np.empty([n_clusters,len(clusters_ventana)-1], dtype=set)
for i in xrange(n_clusters):
    for j in xrange(len(clusters_ventana)-1):
        matriz_freq[i,j]=set(freq[i][j])
        matriz_len[i,j]=len(matriz_freq[i,j])

#Hacer una funcion en donde dentro vayas moviendo las ks con un for pero que le vayas insertando la k, k-esima.
for k in xrange(n_clusters):
    matriz_freq=criss_cross(matriz_freq,matriz_len,k)

#Sigue checar que sets son nulos y eliminar esas ventanas
clusters.columns=list(xrange(clusters.shape[1]))
new_len=pd.DataFrame(matriz_len)
for j in xrange(matriz_freq.shape[1]):
    for k in xrange(matriz_freq.shape[0]):
        new_len.iloc[k][j]=len(matriz_freq[k,j])
suma=new_len.sum()
for j in xrange(len(suma)):
    if suma[j]==0:
        clusters=clusters.drop(j+1,axis=1)


#Ahora falta cambiar los valores de las k en las ventanas j=1 hasta num_ventanas por las k a las que se parecen:

clusters_original=clusters.copy()
for j in clusters.columns:
    for k in xrange(n_clusters):
        for i in xrange(len(clusters)):
            similar_k=list(matriz_freq[k,j-1])
            similar_clusters=clusters_original.ix[:,j].isin(similar_k)
            if similar_clusters[i]==True:
                clusters.iloc[i][j]=k

clusters_moda=list()
for i in xrange(len(clusters)):
    temp=Counter(clusters.ix[i,:])
    clusters_moda.append(temp.most_common(1))
    clusters_moda[i]=clusters_moda[i][0][0]
temp=df
temp=temp.drop('Fecha')
temp=temp.drop(xrange(1,df.shape[1]+1),axis=1)

modas=pd.DataFrame(clusters_moda,index=temp.index, columns=['Cluster'])


# error_df.to_csv('errores_3K')
# clusters.to_csv('KMeans_3K')
