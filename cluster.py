import pandas
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage  


FILENAME = 'suicides.csv'
CLUSTERS = 10

#  lê o csv
dataFrame = pandas.read_csv(FILENAME)

# normaliza oss dados de categorico para numerico
dataFrameNormalized = preprocessing.OrdinalEncoder().fit(dataFrame.values).transform(dataFrame.values)
# print(dataFrameNormalized)

# faz a clusterização hierarquica
cluster = AgglomerativeClustering(n_clusters=CLUSTERS, affinity='euclidean', linkage='ward')
cluster.fit_predict(dataFrameNormalized)

print(len(cluster.labels_))

# plt.plot(clustering.labels_)
# plt.ylabel('Fitness')
# plt.xlabel('Generation')
# plt.show()

plt.scatter(dataFrame.values[:,0],dataFrame.values[:,1], c=cluster.labels_, cmap='rainbow')  