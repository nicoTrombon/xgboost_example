# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import os

os.system('cls' if os.name == 'nt' else 'clear')

# load data
dataset = loadtxt('data/pima-indians-diabetes.data.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

# split data into train and test sets
seed = 7  # set random seed for reproducible results
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model on training data
print 'XGBoost:'
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
auc_score = roc_auc_score(y_test, predictions)
print("Area under ROC curve: %.2f%%" % (auc_score * 100.0))

# Compare with early stopping
print '\nXGBoost with early stopping:'
eval_set = [(X_test, y_test)]
model2 = XGBClassifier()
model2.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

# make predictions for test data
y_pred = model2.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
auc_score = roc_auc_score(y_test, predictions)
print("Area under ROC curve: %.2f%%" % (auc_score * 100.0))

# compare with SVC (without parameter tuning)
print '\nSVC without parameter tuning:'
clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
auc_score = roc_auc_score(y_test, predictions)
print("Area under ROC curve: %.2f%%" % (auc_score * 100.0))

# compare with SVC (with parameter tuning)
print '\nSVC with parameter tuning:'
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
clf = GridSearchCV(SVC(C=1), param_grid, cv=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
auc_score = roc_auc_score(y_test, predictions)
print("Area under ROC curve: %.2f%%" % (auc_score * 100.0))


# compare with Logistic Regression
print '\nLogistic Regression without parameter tuning:'
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
auc_score = roc_auc_score(y_test, predictions)
print("Area under ROC curve: %.2f%%" % (auc_score * 100.0))
