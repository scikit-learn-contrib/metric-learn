"""
Choosing a decision threshold for pairs
=======================================

ITML learns a distance from labeled pairs. To predict whether two samples
match, it compares their learned distance with a threshold. Calibrating
that threshold changes the predictions while keeping the learned distance
fixed.

Here, two Iris samples match if they belong to the same species. Accuracy
treats all prediction errors equally. An F-score with beta greater than
one gives recall more weight. When accepting the wrong match is costly,
we can instead maximize recall subject to a minimum true negative rate.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from metric_learn import ITML


def make_pairs(features, labels, indices):
  """Form pairs and assign +1 to matches and -1 to nonmatches."""
  first, second = np.triu_indices(len(indices), k=1)
  first, second = indices[first], indices[second]
  pairs = features[np.column_stack((first, second))]
  return pairs, np.where(labels[first] == labels[second], 1, -1)


# %%
# Split observations before forming pairs so that no flower appears in more
# than one split. Fit on training pairs, choose thresholds on validation
# pairs, and report performance on test pairs. A small training split keeps
# the number of training pairs manageable.
X, y = load_iris(return_X_y=True)
train, held_out = train_test_split(
    np.arange(len(y)), train_size=0.2, stratify=y, random_state=0)
valid, test = train_test_split(held_out, test_size=0.5, stratify=y[held_out],
                               random_state=0)
pairs_train, y_train = make_pairs(X, y, train)
pairs_valid, y_valid = make_pairs(X, y, valid)
pairs_test, y_test = make_pairs(X, y, test)

learner = ITML().fit(pairs_train, y_train)

# %%
# Recalibrate the same fitted learner for each objective. The minimum true
# negative rate applies to the validation set. The test set may achieve a
# different rate. TPR is the fraction of matching pairs accepted, and TNR
# is the fraction of nonmatching pairs rejected.
strategies = (
    ('Accuracy', {'strategy': 'accuracy'}),
    ('F2', {'strategy': 'f_beta', 'beta': 2}),
    ('Recall with TNR >= 0.95', {'strategy': 'max_tpr', 'min_rate': 0.95}),
)
print('{:<24} {:>9} {:>11} {:>9} {:>9}'.format(
    'Objective', 'Threshold', 'Valid TNR', 'Test TPR', 'Test TNR'))
for title, options in strategies:
  learner.calibrate_threshold(pairs_valid, y_valid, **options)
  tn_valid, fp_valid, _, _ = confusion_matrix(
      y_valid, learner.predict(pairs_valid), labels=[-1, 1]).ravel()
  tn, fp, fn, tp = confusion_matrix(
      y_test, learner.predict(pairs_test), labels=[-1, 1]).ravel()
  print('{:<24} {:>9.3f} {:>11.3f} {:>9.3f} {:>9.3f}'.format(
      title, learner.threshold_, tn_valid / (tn_valid + fp_valid),
      tp / (tp + fn), tn / (tn + fp)))
