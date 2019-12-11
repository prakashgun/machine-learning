import numpy as np
import matplotlib.pyplot as plt

from matplotlib import style

style.use('ggplot')
from sklearn.cluster import KMeans

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

plt.scatter(x, y)
plt.savefig('graph.png')

p = np.array([l for l in zip(x, y)])

kmeans = KMeans(n_clusters=2)
kmeans.fit(p)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

colors = ['g.', 'r.', 'c.', 'y.']

for i in range(len(p)):
    print("Coordinate: ", p[i], ', Label:', labels[i])
    plt.plot(p[i][0], p[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5, zorder=10)

plt.savefig('graph.png')
