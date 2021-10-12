"""
Grid serach use case
=============

Grid search for parameter C in OASIS algorithm
"""

from metric_learn.oasis import OASIS
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state
from sklearn.model_selection import cross_val_score
import numpy as np
from metric_learn.constraints import Constraints
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

SEED = 33
RNG = check_random_state(SEED)

# Load Iris
X, y = load_iris(return_X_y=True)

# Generate triplets
constraints = Constraints(y)
k_geniuine = 3
k_impostor = 10
triplets = constraints.generate_knntriplets(X, k_geniuine, k_impostor)
triplets = X[triplets]

# Values to test for c, folds, and estimator
Cs = np.logspace(-8, 1, 20)
folds = 6  # Cross-validations folds
oasis = OASIS(random_state=RNG)


def find_best_and_plot(plot=True, verbose=True, cv=5):
  """
  Performs a manual grid search of parameter c, then plots
  the cross validation score for each value of c.

  Returns the best score, and the value of c for that score.

  plot: If True will plot a Score vs value of C chart
  verbose: If True will tell in wich iteration it goes.
  cv: Number of cross-validation folds.
  """
  # Save the cross val results of each c
  scores = list()
  scores_std = list()
  c_list = list()
  i = 0
  for c in Cs:
    if verbose:
      print(f'Evaluating param # {i} | c={c}')
    oasis.c = c  # Change c each time
    this_scores = cross_val_score(oasis, triplets, n_jobs=-1, cv=cv)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
    c_list.append(c)
    i = i + 1

  # Plot the cross_val_scores
  if plot:
    plt.figure()
    plt.semilogx(Cs, scores)
    plt.semilogx(Cs, np.array(scores) + np.array(scores_std), 'b--')
    plt.semilogx(Cs, np.array(scores) - np.array(scores_std), 'b--')
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
    plt.ylabel('OASIS score')
    plt.xlabel('Parameter C')
    plt.ylim(0, 1.1)
    plt.show()

  return scores[np.argmax(scores)], c_list[np.argmax(scores)]


def grid_serach(cv=5, verbose=1):
  """
  Performs grid serach using sklearn's GridSearchCV.
  verbose: If True will tell in wich iteration it goes.

  Returns the best score, and the value of c for that score.

  cv: Number of cross-validation folds.
  verbose: Controls the prints of GridSearchCV
  """
  clf = GridSearchCV(estimator=oasis,
                     param_grid=dict(c=Cs), n_jobs=-1, cv=cv,
                     verbose=verbose)
  clf.fit(triplets)
  return clf.best_score_, clf.best_estimator_.c


# Both manual serach and GridSearchCV should output the same value
s1, c1 = find_best_and_plot(plot=True, verbose=True, cv=folds)
s2, c2 = grid_serach(cv=folds, verbose=1)

results = f"""
Manual search
-------------
Best score: {s1}
Best c:     {c1}


GridSearchCV
------------
Best score: {s2}
Best c:     {c2}"""

print(results)
