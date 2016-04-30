__author__ = 'raul'

#Importamos los modulos
from matplotlib.finance import quotes_historical_yahoo as yahoo #Es de donde se obtienen los datos
from datetime import date #Aparentemente necesitas este tipo de objeto para yahoo
from math import log #Se utiliza para calcular los rendimientos logaritmicos
from itertools import repeat #Funciona muy bien para inicializar listas de n listas
import pandas as pd
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
    for i in xrange(len(yahoo_ticker)):
        for j in xrange(1,len(yahoo_object[0])):
            appending_data=log(yahoo_object[i][j][4]/yahoo_object[i][j-1][4])
            rend_log[i].append(appending_data)
            appending_date=date.fromordinal(int(yahoo_object[i][j][0])) #Se cambia la fecha ordinal (gregoriana
                                                                        # proleptica) a gregoriana
            gregorian_date[i].append(appending_date)
            appending_month=appending_date.month
            months[i].append(appending_month)
    df_rend_log=pd.DataFrame(rend_log)
    df_greg_date=pd.DataFrame(gregorian_date)
    month_df=pd.DataFrame(months)
    return df_rend_log, df_greg_date, month_df

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

#Se definen las fechas de inicio y fin
start_date=date(2014, 1, 1)
end_date=date(2016, 2, 28)

#Ultimos activos que vimos en su prospecto de inversion:
yahoo_ticker=['AC.MX', 'ALFAA.MX', 'ALPEKA.MX', 'ALSEA.MX', 'AMXL.MX', 'ASURB.MX', 'BIMBOA.MX', 'BOLSAA.MX',
        'COMERCIUBC.MX', 'ELEKTRA.MX', 'GAPB.MX', 'GENTERA.MX', 'GFINBURO.MX', 'GFNORTEO.MX', 'GFREGIOO.MX',
        'GMEXICOB.MX', 'GRUMAB.MX', 'GSANBORB-1.MX', 'ICA.MX', 'ICHB.MX', 'IENOVA.MX', 'KIMBERA.MX', 'KOFL.MX',
        'LABB.MX', 'LALAB.MX', 'LIVEPOLC-1.MX', 'MEXCHEM.MX', 'OHLMEX.MX', 'PINFRA.MX', 'SANMEXB.MX', 'TLEVISACPO.MX',
        'WALMEX.MX']

#Creamos el df:
df=pd.DataFrame(columns=['ticker', 'yahoo_ticker'])
df['yahoo_ticker']=yahoo_ticker
csv_file=pd.read_csv('NAICS.csv')
df['ticker']=csv_file['ticker']
df['industry']=csv_file['Main Activities']

#Bajamos los datos utilizando yahoo y guardandolos en yahoo_object:
yahoo_object=download_data(yahoo_ticker, start_date, end_date)

#Se obtiene el rendimiento logaritmico y la fecha ordinal, asi como un dataframe de los meses
rend_log , gregorian_date, month_df=log_return(yahoo_ticker,yahoo_object)

#Se obtiene una lista con los dataframes separados en ventanas
ventana=windows(month_df,rend_log,start_date,end_date)

#Redes Neuronales Competitivas basandome en el algoritmo que vi con Riemann:

#--------------Entrenamiento------------#
num_var=list()
weights=list()
num_neuronas=5
for i in xrange(len(ventana)):
    num_var.append(ventana[i].shape[1])
    weights.append(np.random.rand(num_var[i],num_neuronas))

num_datos=ventana[0].shape[0]

epocas=100
eta0=1
delta_eta=eta0/epocas

for epoca in xrange(epocas):
    dist_euclidiana=np.zeros((1,num_neuronas))
    eta=eta0-delta_eta*(epoca-1)
    for i in xrange(len(ventana)):
        datos=ventana[i].reindex(columns=np.random.permutation(ventana[i].columns))
        for j in xrange(num_datos):
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
    for j in xrange(num_datos):
        x_vector=ventana[i].iloc[j][:]
        for neurona in xrange(num_neuronas):
            error=x_vector.values-weights[i][:,neurona]
            dist_euclidiana[0,neurona]=np.sqrt(np.dot(error,error))
        dist_euclidiana_list=list(dist_euclidiana[0])
        indice=dist_euclidiana_list.index(min(dist_euclidiana_list))
        y_vectors[i].append(indice)
# for i in xrange(len(y_vectors)):
#     print(np.unique(y_vectors[i]))
