import random
import numpy as np
from six.moves import xrange

class Constraints(object):

  @staticmethod
  def adjacencyMatrix(labels, num_points, num_constraints):
    a, c = np.random.randint(len(labels), size=(2,num_constraints))
    b, d = np.empty((2, num_constraints), dtype=int)
    for i,(al,cl) in enumerate(zip(labels[a],labels[c])):
      b[i] = random.choice(np.nonzero(labels == al)[0])
      d[i] = random.choice(np.nonzero(labels != cl)[0])
    W = np.zeros((num_points,num_points))
    W[a,b] = 1
    W[c,d] = -1
    # make W symmetric
    W[b,a] = 1
    W[d,c] = -1
    return W

  @staticmethod
  def positiveNegativePairs(labels, num_points, num_constraints):
    ac,bd = np.random.randint(num_points, size=(2,num_constraints))
    pos = labels[ac] == labels[bd]
    a,c = ac[pos], ac[~pos]
    b,d = bd[pos], bd[~pos]
    return a,b,c,d

  @staticmethod
  def relativeQuadruplets(labels, num_constraints):
    C = np.empty((num_constraints,4), dtype=int)
    a, c = np.random.randint(len(labels), size=(2,num_constraints))
    for i,(al,cl) in enumerate(zip(labels[a],labels[c])):
      C[i,1] = random.choice(np.nonzero(labels == al)[0])
      C[i,3] = random.choice(np.nonzero(labels != cl)[0])
    C[:,0] = a
    C[:,2] = c
    return C

  @staticmethod
  def chunks(Y, num_chunks=100, chunk_size=2, seed=None):
    random.seed(seed)
    chunks = -np.ones_like(Y, dtype=int)
    uniq, lookup = np.unique(Y, return_inverse=True)
    all_inds = [set(np.where(lookup==c)[0]) for c in xrange(len(uniq))]
    idx = 0
    while idx < num_chunks and all_inds:
      c = random.randint(0, len(all_inds)-1)
      inds = all_inds[c]
      if len(inds) < chunk_size:
        del all_inds[c]
        continue
      ii = random.sample(inds, chunk_size)
      inds.difference_update(ii)
      chunks[ii] = idx
      idx += 1
    if idx < num_chunks:
      raise ValueError('Unable to make %d chunks of %d examples each' %
                       (num_chunks, chunk_size))
    return chunks
