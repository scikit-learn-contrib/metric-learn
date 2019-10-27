from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from metric_learn import LMNN

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LMNN(verbose=True)
model.fit(X_train, y_train)
