from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

import numpy as np

import pandas


FILENAME = 'suicides.csv'
CLUSTERS = 10

#  lê o csv
dataFrame = pandas.read_csv('csv/' + FILENAME)

# remove as colunas nao usadas
dataFrame = dataFrame.drop(columns=['ano', 'municipio', 'raca', 'microregiao'])

# print(dataFrame.columns.values)

def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# normaliza oss dados de categorico para numerico
data = preprocessing.OrdinalEncoder().fit(dataFrame.values).transform(dataFrame.values)

# # faz a clusterização hierarquica
model = AgglomerativeClustering(
    compute_full_tree=True, 
    linkage="complete", 
    affinity="euclidean", 
    n_clusters=CLUSTERS
)

model.fit(data)

print(model.n_leaves_)

dataFrame['cluster'] = model.labels_

print(dataFrame.groupby(dataFrame['cluster'],as_index=False).size())

# plt.title('Hierarchical Clustering Dendrogram')
# plot_dendrogram(model, labels=model.labels_)
# plt.show()
