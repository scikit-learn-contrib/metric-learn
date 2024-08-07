from metric_learn import LMNN, NCA, LFDA, MLKR, RCA_Supervised, ITML_Supervised, LSML_Supervised, MMC_Supervised, SDML_Supervised, SCML_Supervised
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from time import time
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

SEED = 33
RNG = check_random_state(SEED)

# #############################################################################
# Load the dataset

lfw_people = fetch_lfw_people(min_faces_per_person=40, resize=0.5)

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
# Split into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
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
# Get a baseline score without using metric learning
clf = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
bs = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
print("{:<15} {:<10} {:<10} {:<10}".format('','Precision','Recall','F-beta'))
print("{:<15} {:<10} {:<10} {:<10}".format('Baseline', str(round(bs[0], 2)), str(round(bs[1], 2)), str(round(bs[2], 2))))

# #############################################################################
# For each model, we train and learn a transformation, then we evaluate on KNN
# Consider tunning the params of each metric learner beforehand

models = [LMNN(random_state=RNG),
          NCA(random_state=RNG),
          LFDA(),
          MLKR(random_state=RNG),
          RCA_Supervised(random_state=RNG),
          ITML_Supervised(random_state=RNG),
          LSML_Supervised(random_state=RNG),
          MMC_Supervised(random_state=RNG),
          SDML_Supervised(random_state=RNG),
          # SCML_Supervised(random_state=RNG)
          ]

pipes = []

for model in models:
  pipes.append(Pipeline([('metric-learner', model), ('knn', KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree'))]))

scores = []

for index, pipe in enumerate(pipes):
  name = models[index].__class__.__name__
  pipe.fit(X_train_pca, y_train)
  y_pred = pipe.predict(X_test_pca)
  s = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
  scores.append(s)
  print("{:<15} {:<10} {:<10} {:<10}".format(name, str(round(s[0], 2)), str(round(s[1], 2)), str(round(s[2], 2))))

# #############################################################################
# Plot precision, Recall, F-beta for each metric learner compared to the baseline
x = [ m.__class__.__name__ for m in models]
f1 = [a[2] for a in scores]

plt.axhline(y=bs[2], color='k', linestyle='-') # Baseline
plt.axhline(y=max(f1), color='r', linestyle='dotted') # Max score

plt.bar(x, f1)
plt.ylabel('F-beta score')
plt.xlabel('Metric learning models');
plt.title('F-beta score for different metric learners')
plt.show()