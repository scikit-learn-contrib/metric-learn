"""
Bilinear similarity example
=============

Bilinear similarity example using OASIS algorithm
"""

from metric_learn import SCML, LMNN, NCA, OASIS, LFDA, MLKR, MMC
from sklearn.datasets import load_iris
from sklearn.utils import check_random_state
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np
from metric_learn.constraints import Constraints, wrap_pairs
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from time import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

SEED = 33
RNG = check_random_state(SEED)


lfw_people = fetch_lfw_people(min_faces_per_person=100, resize=0.5)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=12
)

# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 50

print(
    "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
)
t0 = time()
pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

# #############################################################################
# Make triplets
print("Building triplets from supervised dataset")
t0 = time()
constraints = Constraints(y_train)
k_geniuine = 3
k_impostor = 4
triplets = constraints.generate_knntriplets(X_train_pca, k_geniuine, k_impostor)
print("done in %0.3fs" % (time() - t0))

# #############################################################################
# OASIS: Values to test for c, folds, and estimator
if False:
  print("Training OASIS model")
  t0 = time()
  oasis = OASIS(random_state=33, preprocessor=X_train_pca, c=0.00162)
  oasis.fit(triplets)
  custom_metric = lambda a, b : - + 1.0 / oasis.get_metric()(a, b)
  print(oasis.score(triplets))

  constraints = Constraints(y_test)
  k_geniuine = 10
  k_impostor = 10
  triplets_test = constraints.generate_knntriplets(X_test_pca, k_geniuine, k_impostor)

  print(oasis.score(triplets_test))
  # print(custom_metric(X_train_pca[0], X_train_pca[0]))
  # print(oasis.get_metric()(X_train_pca[0], X_train_pca[0]))
  # # print(oasis.get_bilinear_matrix().min())
  # print(custom_metric)
  
  # Tunning OASIS

  # Cs = np.logspace(-8, 1, 20)
  # folds = 4  # Cross-validations folds
  # clf = GridSearchCV(estimator=oasis,
  #                   param_grid=dict(c=Cs), n_jobs=-1, cv=folds,
  #                   verbose=True)
  # clf.fit(triplets)
  # print(f"Best c: {clf.best_estimator_.c}")
  # print(f"Best score: {clf.best_score_}")
  # custom_metric = clf.best_estimator_.get_metric()
  # print("done in %0.3fs" % (time() - t0))

# #############################################################################
if False:
  print("Training SCML model")
  scml = SCML(random_state=33, preprocessor=X_train_pca)
  scml.fit(triplets)
  custom_metric = scml.get_metric()
  print("done in %0.3fs" % (time() - t0))
  print(scml.score(triplets))

  constraints = Constraints(y_test)
  k_geniuine = 3
  k_impostor = 4
  triplets_test = constraints.generate_knntriplets(X_test_pca, k_geniuine, k_impostor)
  print(scml.score(triplets_test))

if True:
  c = Constraints(y_train)
  p = c.positive_negative_pairs(1000)
  pairs, label = wrap_pairs(X_train_pca, p)

  mmc = MMC(random_state=22)
  mmc.fit(pairs, label)
  print(mmc.score(pairs, label))

  c1 = Constraints(y_test)
  p1 = c1.positive_negative_pairs(1000)
  pairs1, label1 = wrap_pairs(X_train_pca, p1)
  print(mmc.score(pairs1, label1))

# #############################################################################
if False:
  print("Training LMNN model")
  lmnn = LMNN(random_state=33, preprocessor=X_train_pca)
  lmnn.fit(X_train_pca, y_train)
  custom_metric = lmnn.get_metric()
  print("done in %0.3fs" % (time() - t0))

# #############################################################################
if False:
  print("Training NCA model")
  nca = NCA(random_state=33, preprocessor=X_train_pca, max_iter=1000)
  nca.fit(X_train_pca, y_train)
  custom_metric = nca.get_metric()
  print("done in %0.3fs" % (time() - t0))

# #############################################################################
if False:
  print("Training MLKR model")
  mlkr = MLKR(preprocessor=X_train_pca, random_state=33)
  mlkr.fit(X_train_pca, y_train)
  custom_metric = mlkr.get_metric()
  print("done in %0.3fs" % (time() - t0))

# #############################################################################
if False:
  print("Training LFDA model")
  lfda = LFDA(preprocessor=X_train_pca)
  lfda.fit(X_train_pca, y_train)
  custom_metric = lfda.get_metric()
  print("done in %0.3fs" % (time() - t0))

# #############################################################################
# KNN Classifier
# print("Fitting Classifier")
# t0 = time()
# neigh = KNeighborsClassifier(n_neighbors=5, metric=custom_metric, algorithm='brute')
# neigh.fit(X_train_pca, y_train)
# print("done in %0.3fs" % (time() - t0))

# # #############################################################################
# # Quantitative evaluation of the model quality on the test set

# print("Predicting people's names on the test set")
# t0 = time()
# y_pred = neigh.predict(X_test_pca)
# print("done in %0.3fs" % (time() - t0))

# print(classification_report(y_test, y_pred, target_names=target_names))
# #print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))