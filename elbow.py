
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import random as rd
import pandas
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

def plotDistortion(data):

    k_range = range(1,50)

    distortions = []

    for i in k_range:
        model = KMeans(n_clusters=i)
        model.fit(data)
        distortions.append(sum(np.min(cdist(data, model.cluster_centers_, 'euclidean'), axis=1)) / data.shape[0])

    # fig1 = plt.figure()
    # ex = fig1.add_subplot(111)
    # ex.plot(k_range, distortions, '*')
    plt.plot(distortions)

    plt.grid(True)

    # plt.ylim([0, 45])
    plt.xlabel('Numero de clusters')
    plt.ylabel('Distorção média')
    plt.title('Seleção de k com o método Elbow')
    plt.show()