import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
iris = load_iris()
# describe dataset
print(iris.DESCR)
# Filter the data for two specific classes
X = iris.data
y = iris.target  # Fix: Assign iris.target to y instead of iris

# divide dataset into training and testing sets with a ratio of 80/20
features_train, features_test, targets_train, targets_test = train_test_split(X,y, test_size=0.2)

# create model
model = LogisticRegression()
model.fit(features_train, targets_train)
prediction = model.predict(features_test)
# calculate the accuracy of the trained model on the testing set
accuracy = accuracy_score(targets_test, prediction)  # Fix: Use targets_test instead of targets_train
print("Accuracy of the Logistic Regression:", accuracy)
