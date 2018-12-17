# -*- coding: utf-8 -*-
"""
Learning on pairs
=========================
"""

##################################################################################
# Let's import a dataset of pairs of images from scikit-learn.

from sklearn.datasets import fetch_lfw_pairs
from sklearn.utils import shuffle

dataset = fetch_lfw_pairs()
pairs, y = shuffle(dataset.pairs, dataset.target, random_state=42)
y = 2*y - 1  # we want +1 to indicate similar pairs and -1 dissimilar pairs

######################################################################################
# Let's print a pair of dissimilar points:

import matplotlib.pyplot as plt
import numpy as np

label = -1
first_pair_idx = np.where(y==label)[0][0]
fig, ax = plt.subplots(ncols=2, nrows=1)
for i, img in enumerate(pairs[first_pair_idx]):
    ax[i].imshow(img, cmap='Greys_r')
fig.suptitle('Pair n°{}, Label: {}\n\n'.format(first_pair_idx, label))
######################################################################################
# Now let's print a pair of similar points:

label = 1
first_pair_idx = np.where(y==label)[0][0]
fig, ax = plt.subplots(ncols=2, nrows=1)
for i, img in enumerate(pairs[first_pair_idx]):
    ax[i].imshow(img, cmap='Greys_r')
fig.suptitle('Pair n°{}, Label: {}\n\n'.format(first_pair_idx, label))
###############################################################################
# Let's reshape the dataset so that it si indeed a 3D array of size ``(n_tuples, 2, n_features)``,
# and print the first three elements


pairs = pairs.reshape(pairs.shape[0], 2, -1)
print(pairs[:3])