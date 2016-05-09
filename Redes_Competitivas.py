__author__ = 'raul'

#Importamos los modulos
from collections import Counter #Para encontrar la moda
from math import log #Se utiliza para calcular los rendimientos logaritmicos
from itertools import repeat #Funciona muy bien para inicializar listas de n listas
import pandas as pd
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

# Creamos el df:
industria=pd.DataFrame(columns=['ticker'])
csv_file=pd.read_csv('NAICS2.csv')
industria['ticker']=csv_file['ticker']
industria['industry']=csv_file['Main Activities']

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

#Redes Neuronales Competitivas basandome en el algoritmo que vi con Riemann:

#--------------Entrenamiento------------#
num_activos=ventana[0].shape[0]
weights=list()
num_neuronas=18
for i in xrange(len(ventana)):
    weights.append(np.zeros((ventana[i].shape[1], num_neuronas)))

for i in xrange(len(ventana)):
    mean_weights=ventana[i].mean()
    for n in xrange(num_neuronas):
        weights[i][:,n]=mean_weights.values+weights[i][:,n]

epocas=500

for epoca in xrange(epocas):
    dist_euclidiana=np.zeros((1,num_neuronas))
    for i in xrange(len(ventana)):
        eta0=ventana[i].mean().mean()
        delta_eta=eta0/epocas
        eta=eta0-delta_eta*(epoca-1)
        datos=ventana[i].reindex(columns=np.random.permutation(ventana[i].columns))
        for j in xrange(num_activos):
            x_vector=datos.iloc[j][:]
            for neurona in xrange(num_neuronas):
                error=x_vector.values-weights[i][:,neurona]
                dist_euclidiana[0,neurona]=np.sqrt(np.dot(error,error))
            dist_euclidiana_list=list(dist_euclidiana[0])
            indice=dist_euclidiana_list.index(min(dist_euclidiana_list))
            weights[i][:,indice]=weights[i][:,indice]+eta*(x_vector.values-weights[i][:,indice])

#---------Simulacion-----------#
y_vectors=[ [] for i in repeat(None,len(ventana))]
for i in xrange(len(ventana)):
    for j in xrange(num_activos):
        x_vector=ventana[i].iloc[j][:]
        for neurona in xrange(num_neuronas):
            error=x_vector.values-weights[i][:,neurona]
            dist_euclidiana[0,neurona]=np.sqrt(np.dot(error,error))
        dist_euclidiana_list=list(dist_euclidiana[0])
        indice=dist_euclidiana_list.index(min(dist_euclidiana_list))
        y_vectors[i].append(indice)
clusters=pd.DataFrame(y_vectors)
clusters=clusters.T
clusters_moda=list()
for i in xrange(len(clusters)):
    temp=Counter(clusters.iloc[i][:])
    clusters_moda.append(temp.most_common(1))
    clusters_moda[i]=clusters_moda[i][0][0]
temp=df
temp=temp.drop('Fecha')
temp=temp.drop(xrange(1,df.shape[1]+1),axis=1)

modas=pd.DataFrame(clusters_moda,index=temp.index, columns=['Cluster'])
modas.to_csv('modas_18n.csv')
clusters.to_csv('RNC_18n.csv')