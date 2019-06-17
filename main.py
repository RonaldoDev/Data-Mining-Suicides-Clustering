from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

import numpy as np

import pandas


FILENAME = 'suicides.csv'
CLUSTERS = 8
#  lê o csv
dataFrame = pandas.read_csv('csv/' + FILENAME)

dataFrame = dataFrame[
    (dataFrame['sexo'] != 'NI') & 
    (dataFrame['estado_civil'] != 'NI') & 
    (dataFrame['escolaridade'] != 'NI') & 
    (dataFrame['IDH'] != 'NI') & 
    (dataFrame['mesoregiao'] != 'NI') &
    (dataFrame['local_cid'] != 'NI') & 
    (dataFrame['sexo'] != 'NI') &
    (dataFrame['Profissao'] != 'NI') &
    (dataFrame['Faixa_etaria'] != 'NI') &
    (dataFrame['motivo'] != 'NI') &
    (dataFrame['local_ocorrencia'] != 'NI') 
]

# remove as colunas nao usadas
dataFrame = dataFrame.drop(columns=['sexo', 'IDH', 'ano', 'local_ocorrencia', 'municipio', 'raca', 'microregiao', 'local_cid'])

# print(dataFrame.columns.values)


encoder = preprocessing.OrdinalEncoder()

# normaliza oss dados de categorico para numerico
dataNumeric = encoder.fit(dataFrame.values).transform(dataFrame.values)
# data = dataFrame

scaler = preprocessing.MinMaxScaler()

data = scaler.fit_transform(dataNumeric)


# # faz a clusterização hierarquica
# model = AgglomerativeClustering(
#     compute_full_tree=True, 
#     linkage="complete", 
#     affinity="euclidean", 
#     n_clusters=CLUSTERS
# )

model = KMeans(n_clusters=CLUSTERS)

model.fit(data)

clusters = model.labels_

centroidsScaler = scaler.inverse_transform(model.cluster_centers_)

centroids = encoder.inverse_transform(centroidsScaler)

df_centroids = pandas.DataFrame(centroids, columns=dataFrame.columns.values)

df_centroids.drop(columns='motivo')

print(df_centroids)

dataFrame['cluster'] = model.labels_

# print(dataFrame.values)


print(dataFrame.groupby(dataFrame['cluster'],as_index=False).size())

# print(dataFrame)

print(model.inertia_)

# print(model.labels_)

# plt.title('Hierarchical Clustering Dendrogram')
# plot_dendrogram(model, labels=model.labels_)
# plt.show()
