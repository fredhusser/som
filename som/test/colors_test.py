"""Script for using the SOM classification algorithm
in combination with the Ward agglomerative clustering
operated on the nodes of the Kohonen map.
The U-matrix corresponding to the Kohonen map is also
plotted on the same figure.
"""

__author__ = 'husser'

import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from som.som import SOMMapper, build_U_matrix

kshape_test = (30, 20)
n_iter_test = 300
learning_rate_test = 0.005
n_colors = 200

spcolors = np.random.rand(n_colors, 3)
mapper = SOMMapper(kshape=kshape_test, n_iter=n_iter_test, learning_rate=learning_rate_test)
kohonen = mapper.fit_transform(spcolors)
U_Matrix = build_U_matrix(kohonen, kshape_test, topology="rect")

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(np.split(kohonen, kshape_test[0], axis=0))
ax1.set_title("Kohonen Map")

# Clustering
n_clusters = 6  # number of regions
connectivity = grid_to_graph(kshape_test[0], kshape_test[1])
ward = AgglomerativeClustering(n_clusters=n_clusters,
                               linkage='ward', connectivity=connectivity).fit(kohonen)

label = np.reshape(ward.labels_, kshape_test)
for l in range(n_clusters):
    ax1.contour(label == l, contours=1,
                colors=[plt.cm.spectral(l / float(n_clusters)), ])

ax2 = fig.add_subplot(122)
im = ax2.imshow(np.split(U_Matrix, kshape_test[0], axis=0))
for l in range(n_clusters):
    ax2.contour(label == l, contours=1,
                colors=[plt.cm.spectral(l / float(n_clusters)), ])

ax2.set_title("U_Matrix")
fig.colorbar(im, ax=ax2)
plt.show()