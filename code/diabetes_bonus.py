from sklearn.datasets import load_diabetes
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_diabetes()
features = iris.data
y = iris.target

# shuffle the dataset
randNum = np.arange(features .shape[0])
np.random.shuffle(randNum)
features = features[randNum]
targets = y[randNum]

# divide dataset into training and testing sets
feature_train, feature_test, target_train, target_test = train_test_split(features, targets, test_size=0.2)

# Add a column of ones to the feature matrix for the bias term
X_train = np.hstack((np.ones((feature_train.shape[0], 1)), feature_train))

# Calculate the manual inverse matrix
XtX_inv = np.linalg.inv(np.dot(X_train.T, X_train))
Xty = np.dot(X_train.T, target_train)

# Calculate the weights using the manual inverse matrix
weights = np.dot(XtX_inv, Xty)

# Print the weights
print("Weights:")
print(weights)

# Predict on the training set
y_pred_train = np.dot(X_train, weights)

# Predict on the testing set
X_test = np.hstack((np.ones((feature_test.shape[0], 1)), feature_test))
y_pred_test = np.dot(X_test, weights)

# calculate mean squared error
print("MSE: " ,mean_squared_error(target_test, y_pred_test))
