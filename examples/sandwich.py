"""
Sandwich demo based on code from http://nbviewer.ipython.org/6576096
"""

import numpy as np
from numpy.random import normal
import matplotlib.pyplot as pyplot
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from metric_learn import ITML, LMNN, LSML, SDML


def sandwich_demo():
  x, y = sandwich_data()
  knn = nearest_neighbors(x, k=2)
  ax = pyplot.subplot(3, 1, 1)  # take the whole top row
  plot_sandwich_data(x, y, ax)
  plot_neighborhood_graph(x, knn, y, ax)
  ax.set_title('input space')
  ax.set_aspect('equal')
  ax.set_xticks([])
  ax.set_yticks([])

  num_constraints = 60
  mls = [
      LMNN(x, y),
      ITML(x, ITML.prepare_constraints(y, len(x), num_constraints)),
      SDML(x, SDML.prepare_constraints(y, len(x), num_constraints)),
      LSML(x, LSML.prepare_constraints(y, num_constraints))
  ]

  for ax_num, ml in zip(xrange(3,7), mls):
    ml.fit()
    tx = ml.transform()
    ml_knn = nearest_neighbors(tx, k=2)
    ax = pyplot.subplot(3,2,ax_num)
    plot_sandwich_data(tx, y, ax)
    plot_neighborhood_graph(tx, ml_knn, y, ax)
    ax.set_title('%s space' % ml.__class__.__name__)
    ax.set_xticks([])
    ax.set_yticks([])
  pyplot.show()


# TODO: use this somewhere
def visualize_class_separation(X, labels):
  _, (ax1,ax2) = pyplot.subplots(ncols=2)
  label_order = np.argsort(labels)
  ax1.imshow(pairwise_distances(X[label_order]), interpolation='nearest')
  ax2.imshow(pairwise_distances(labels[label_order,None]),
             interpolation='nearest')
  pyplot.show()


def nearest_neighbors(X, k=5):
  knn = NearestNeighbors(n_neighbors=k)
  knn.fit(X)
  return knn.kneighbors(X, return_distance=False)


def sandwich_data():
  # number of distinct classes
  num_classes = 6
  # number of points per class
  num_points = 9
  # distance between layers, the points of each class are in a layer
  dist = 0.7
  # memory pre-allocation
  x = np.zeros((num_classes*num_points, 2))
  y = np.zeros(num_classes*num_points, dtype=int)
  for i,j in zip(xrange(num_classes), xrange(-num_classes//2,num_classes//2+1)):
    for k,l in zip(xrange(num_points), xrange(-num_points//2,num_points//2+1)):
      x[i*num_points + k, :] = np.array([normal(l, 0.1), normal(dist*j, 0.1)])
    y[i*num_points:i*num_points + num_points] = i
  return x,y


def plot_sandwich_data(x, y, axis=pyplot, cols='rbgmky'):
  for idx,val in enumerate(np.unique(y)):
    xi = x[y==val]
    axis.scatter(xi[:,0], xi[:,1], s=50, facecolors='none',edgecolors=cols[idx])


def plot_neighborhood_graph(x, nn, y, axis=pyplot, cols='rbgmky'):
  for i in xrange(x.shape[0]):
    xs = [x[i,0], x[nn[i,1], 0]]
    ys = [x[i,1], x[nn[i,1], 1]]
    axis.plot(xs, ys, cols[y[i]])


if __name__ == '__main__':
  sandwich_demo()
