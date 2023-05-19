import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
# load dataset
iris = load_iris()
# describe dataset
print(iris.DESCR)
# Filter the data for two specific classes
X = iris.data
y = iris.target  # Fix: Assign iris.target to y instead of iris

class1_label = 0
class2_label = 1
class1_samples = X[y == class1_label]
class2_samples = X[y == class2_label]
# Combine the samples from the two classes
filtered_samples = np.concatenate((class1_samples, class2_samples), axis=0)
filtered_labels = np.concatenate((np.zeros(len(class1_samples)), np.ones(len(class2_samples))), axis=0)

# divide dataset into training and testing sets with a ratio of 80/20
features_train, features_test, targets_train, targets_test = train_test_split(filtered_samples, filtered_labels, test_size=0.2)

# create model
model = LogisticRegression()
model.fit(features_train, targets_train)
prediction = model.predict(features_test)

# Train a k-NN model (using Sklearn) with different hyperparameters on the training set.
#  define values for different hyperparameters
n_neighbors_values = [5, 3, 7]
for n_neighbors in n_neighbors_values:
    #  K -NN model (SK learn ) on the training set
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the model using the training set
    knn.fit(features_train, targets_train)

    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)  # Fix: Use targets_test instead of targets_train
    print("Accuracy of the Logistic Regression:", accuracy)
