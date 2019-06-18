from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import random as rd
import pandas


FILENAME = 'suicides.csv'
CLUSTERS = 12


#  lê o csv
dataFrame = pandas.read_csv('csv/' + FILENAME)

# dataFrame = dataFrame[
#     # (dataFrame['sexo'] != 'NI') & 
#     (dataFrame['estado_civil'] != 'NI') & 
#     (dataFrame['escolaridade'] != 'NI') & 
#     # (dataFrame['IDH'] != 'NI') & 
#     (dataFrame['mesoregiao'] != 'NI') &
#     # (dataFrame['local_cid'] != 'NI') & 
#     # (dataFrame['sexo'] != 'NI') &
#     (dataFrame['Profissao'] != 'NI') &
#     (dataFrame['Faixa_etaria'] != 'NI') &
#     (dataFrame['motivo'] != 'NI') 
#     # (dataFrame['local_ocorrencia'] != 'NI') 
# ]

# remove as colunas nao usadas
dataFrame = dataFrame.drop(columns=['sexo', 'idh', 'ano', 'local_ocorrencia', 'municipio', 'raca', 'microregiao'])


dataFrame['estado_civil'] = dataFrame['estado_civil'].replace('NI', dataFrame['estado_civil'][rd.randrange(0, len(dataFrame['estado_civil'].unique()) - 1)])
dataFrame['faixa_etaria'] = dataFrame['faixa_etaria'].replace('NI', dataFrame['faixa_etaria'][rd.randrange(0, len(dataFrame['faixa_etaria'].unique()) - 1)])
dataFrame['escolaridade'] = dataFrame['escolaridade'].replace('NI', dataFrame['escolaridade'][rd.randrange(0, len(dataFrame['escolaridade'].unique()) - 1)])
dataFrame['profissao'] = dataFrame['profissao'].replace('NI', dataFrame['profissao'][rd.randrange(0, len(dataFrame['profissao'].unique()) - 1)])
dataFrame['mesoregiao'] = dataFrame['mesoregiao'].replace('NI', dataFrame['mesoregiao'][rd.randrange(0, len(dataFrame['mesoregiao'].unique()) - 1)])
dataFrame['motivo'] = dataFrame['motivo'].replace('NI', dataFrame['motivo'][rd.randrange(0, len(dataFrame['motivo'].unique()) - 1)])

# export_csv = dataFrame.to_csv (r'C:\Users\daniel.kock\Documents\DM\Data-Mining-Suicides-Clustering\df.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path
# print(dataFrame)

# normaliza oss dados de categorico para numerico
encoder = preprocessing.OrdinalEncoder()

dataNumeric = encoder.fit(dataFrame.values).transform(dataFrame.values)

# normaliza os dados numericos para intervalo de 0 e 1
scaler = preprocessing.MinMaxScaler()

data = scaler.fit_transform(dataNumeric)


# faz a clusterização usando k-means
model = KMeans(n_clusters=CLUSTERS)

# treina o modelo
model.fit(data)

clusters = model.labels_

# converte os centroides para valores categoricos novamente
centroidsScaler = scaler.inverse_transform(model.cluster_centers_)

centroids = encoder.inverse_transform(centroidsScaler)

df_centroids = pandas.DataFrame(centroids, columns=dataFrame.columns.values)


# print(df_centroids)

# dataFrame['cluster'] = model.labels_

# print(dataFrame.values)


# print(dataFrame.groupby(dataFrame['cluster'],as_index=False).size())

# print(dataFrame)

print(model.inertia_)

# print(model.labels_)

# plt.title('Hierarchical Clustering Dendrogram')
# plot_dendrogram(model, labels=model.labels_)
# plt.show()
