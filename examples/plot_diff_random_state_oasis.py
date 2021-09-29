from metric_learn.oasis import OASIS
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state
from sklearn.model_selection import cross_val_score
import numpy as np
from metric_learn.constraints import Constraints
import matplotlib.pyplot as plt

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
rs = np.arange(30)
folds = 6  # Cross-validations folds
c = 0.006203576
oasis = OASIS(c=c, custom_M="random")  # M init random


def random_theory(plot=True, verbose=True, cv=5):
  # Save the cross val results of each c
  scores = list()
  scores_std = list()
  rs_l = list()
  i = 0
  for r in rs:
    oasis.random_state = check_random_state(r)  # Change rs each time
    this_scores = cross_val_score(oasis, triplets, n_jobs=-1, cv=cv)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
    rs_l.append(r)
    if verbose:
      print(f"""Evaluating param # {i} | random_state={r} \
|score: {np.mean(this_scores)}""")
    i = i + 1

  # Plot the cross_val_scores
  if plot:
    plt.figure()
    plt.plot(rs, scores)
    plt.plot(rs, np.array(scores) + np.array(scores_std), 'b--')
    plt.plot(rs, np.array(scores) - np.array(scores_std), 'b--')
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
    plt.ylabel(f'OASIS score with c={c}')
    plt.xlabel('Random State (For shuffling and M init)')
    plt.ylim(0, 1.1)
    plt.show()

  max_i = np.argmax(scores)
  min_i = np.argmin(scores)
  avg = np.average(scores)
  avgstd = np.average(scores_std)
  return scores[max_i], rs_l[max_i], scores[min_i], rs_l[min_i], avg, avgstd


maxs, maxrs, mins, minrs, avg, avgstd = random_theory(cv=folds,
                                                      plot=True,
                                                      verbose=True)

msg = f"""
Max Score     : {maxs}
Max Score Seed: {maxrs}
---------------
Min Score     : {mins}
Min Score Seed: {minrs}
--------------
Average Score : {avg}
Average Std.  : {avgstd}
"""
print(msg)
