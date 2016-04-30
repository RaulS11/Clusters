__author__ = 'raul'

#Importamos los modulos
from matplotlib.finance import quotes_historical_yahoo as yahoo #Es de donde se obtienen los datos
from datetime import date #Aparentemente necesitas este tipo de objeto para yahoo
from math import log #Se utiliza para calcular los rendimientos logaritmicos
from itertools import repeat #Funciona muy bien para inicializar listas de n listas
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

#Se bajan los datos de yahoo finance utilizando matplotlib.finance.quotes_historical_yahoo:
def download_data(yahoo_ticker, start_date, end_date):
    yahoo_object=[]
    for i in xrange(len(yahoo_ticker)):
        appending_data=yahoo(yahoo_ticker[i], start_date, end_date)
        yahoo_object.append(appending_data)
    return yahoo_object

#Se obtiene el rendimiento logaritmico (con respecto al precio de cierre):
def log_return(yahoo_ticker,yahoo_object):
    rend_log=[ [] for i in repeat([],len(yahoo_ticker))] #Se crea un arreglo de tamanho [n]*[Null]
    gregorian_date=[ [] for i in repeat(None,len(yahoo_ticker))]
    months=[ [] for i in repeat(None,len(yahoo_ticker))]
    ordinal_date=[ [] for i in repeat([],len(yahoo_ticker))]
    for i in xrange(len(yahoo_ticker)):
        for j in xrange(1,len(yahoo_object[i])):
            appending_data=log(yahoo_object[i][j][4]/yahoo_object[i][j-1][4])
            rend_log[i].append(appending_data)
            appending_date=date.fromordinal(int(yahoo_object[i][j][0])) #Se cambia la fecha ordinal (gregoriana
                                                                        # proleptica) a gregoriana
            gregorian_date[i].append(appending_date)
            appending_month=appending_date.month
            months[i].append(appending_month)
            appending_date=yahoo_object[i][j][0]
            ordinal_date[i].append(appending_date)
    df_rend_log=pd.DataFrame(rend_log)
    df_rend_log=df_rend_log.dropna(axis=1)
    df_greg_date=pd.DataFrame(gregorian_date)
    df_greg_date=df_greg_date.dropna(axis=1)
    month_df=pd.DataFrame(months)
    month_df=month_df.dropna(axis=1)
    ordinal_date_df=pd.DataFrame(ordinal_date)
    ordinal_date_df=ordinal_date_df.dropna(axis=1)
    return df_rend_log, df_greg_date, month_df, ordinal_date_df

#Se obtienen las ventanas bursatiles
def windows(month_df, rend_log, start_date, end_date):
    #Se obtienen el numero de ventanas bursatiles
    delta_year=end_date.year-start_date.year
    if delta_year > 1:
        num_ventanas= (13-start_date.month) + (delta_year-1)*(12) + end_date.month
    else:
        num_ventanas=end_date.month-start_date.month

    #Se obtienen las ventanas bursatiles
    ventana=[]

    sumas=month_df.sum()
    j=0
    for i in xrange(1,len(sumas)):
        if sumas[i]>sumas[i-1] or (sumas[i]==1*len(rend_log) and sumas[i-1]==12*len(rend_log)):
            ventana.append(rend_log.iloc[:len(rend_log),j:i]) #Se debe de hacer con rend_log pero hare la pruena con
                                                              #month_df
            j=i

    if len(ventana)<num_ventanas:
        ventana.append(rend_log.iloc[:len(rend_log),j:rend_log.shape[1]])
    return ventana

#Se obtiene el cluster y el modelo (mas bien la deje porque si no 'returneo' el modelo de una funcion no puedo acceder
# al atributo de los centroides)
def cluster_data(data, n_clusters=3):
    cluster_model = KMeans(n_clusters)
    prediction = cluster_model.fit_predict(data)
    return prediction, cluster_model

#Continue utilizando el mismo error que en el programa anterior
def measure_error(prediction, model, c_data):
    error_score = []
    for counter in range(len(c_data)):
        true_val = c_data.drop("Cluster",1).values[counter]
        center_val = model.cluster_centers_[c_data["Cluster"][counter]]

        error_score.append(np.average(np.abs(true_val - center_val)) / np.average(center_val))

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

#Se definen las fechas de inicio y fin
start_date=date(2013, 11, 1)
end_date=date(2016, 3, 31)

#Ultimos activos que vimos en su prospecto de inversion:

#Items que nos habia pasado Francisco
# yahoo_ticker=['AC.MX', 'ALFAA.MX', 'ALPEKA.MX', 'ALSEA.MX', 'AMXL.MX', 'ASURB.MX', 'BIMBOA.MX', 'BOLSAA.MX',
#         'COMERCIUBC.MX', 'ELEKTRA.MX', 'GAPB.MX', 'GENTERA.MX', 'GFINBURO.MX', 'GFNORTEO.MX', 'GFREGIOO.MX',
#         'GMEXICOB.MX', 'GRUMAB.MX', 'GSANBORB-1.MX', 'ICA.MX', 'ICHB.MX', 'IENOVA.MX', 'KIMBERA.MX', 'KOFL.MX',
#         'LABB.MX', 'LALAB.MX', 'LIVEPOLC-1.MX', 'MEXCHEM.MX', 'OHLMEX.MX', 'PINFRA.MX', 'SANMEXB.MX', 'TLEVISACPO.MX',
#         'WALMEX.MX']

yahoo_ticker=['AC.MX', 'ALFAA.MX', 'AMX', 'ASURB.MX', 'BIMBOA.MX', 'CEMEXCPO.MX', 'CREAL.MX', 'DANHOS13.MX',
              'ELEKTRA.MX', 'FIHO12.MX', 'GAPB.MX', 'GBMCREBD.MX', 'GENTERA.MX', 'GFINBURO.MX', 'GFNORTEO.MX',
              'GMEXICOB.MX', 'GRUMAB.MX', 'HBCN.MX', 'IENOVA.MX', 'KIMBERA.MX', 'KOFL.MX', 'LABB.MX', 'LIVEPOLC-1.MX',
              'MEXCHEM.MX', 'OMAB.MX', 'PRINFGUFF1.MX', 'RASSINIA.MX', 'SANMEXB.MX', 'TERRA13.MX', 'TLEVISACPO.MX',
              'VESTA.MX', 'WALMEX.MX']

#Creamos el df:
df=pd.DataFrame(columns=['ticker', 'yahoo_ticker'])
df['yahoo_ticker']=yahoo_ticker

##[Hay que volver a sacar las industrias de los que faltan]
#csv_file=pd.read_csv('NAICS.csv')
#df['ticker']=csv_file['ticker']
# df['industry']=csv_file['Main Activities']

#Bajamos los datos utilizando yahoo y guardandolos en yahoo_object:
yahoo_object=download_data(yahoo_ticker, start_date, end_date)

#Se obtiene el rendimiento logaritmico y la fecha ordinal, asi como un dataframe de los meses
rend_log , gregorian_date, month_df, ordinal_date=log_return(yahoo_ticker,yahoo_object)

#Se obtiene una lista con los dataframes separados en ventanas
ventana=windows(ordinal_date,rend_log,start_date,end_date)

#
n_clusters=5
ventana, error_df = clustering(ventana,n_clusters)