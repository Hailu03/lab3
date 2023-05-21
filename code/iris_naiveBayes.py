import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
iris = load_iris()

# Filter the data for two specific classes
X = iris.data
y = iris.target 

# shuffle the dataset
randNum = np.arange(X.shape[0])
np.random.shuffle(randNum)
X = X[randNum]
y = y[randNum]

# divide dataset into training and testing sets with a ratio of 80/20
features_train, features_test, targets_train, targets_test = train_test_split(X,y, test_size=0.2)

# create model
clf = GaussianNB()
clf.fit(features_train, targets_train) # Train the model using the training set

# predict the class labels for the test set
prediction = clf.predict(features_test)

# calculate the accuracy of the trained model on the testing set
accuracy = accuracy_score(targets_test, prediction)
print(f"Accuracy of the Naive Bayes Classifier: {accuracy*100}%")