import numpy as np
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_model(features_train, targets_train,n_neighbors,metric='minkowski',weight='uniform'):
    # create model
    clf = GaussianNB()
    clf.fit(features_train, targets_train) # Train the model using the training set
    return clf

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

# create naivebayes with different parameters
priors = [None,[0.1,0.2,0.3,0.4],[0.2,0.2,0.3,0.3],[0.3,0.3,0.2,0.2],[0.4,0.3,0.2,0.1]]
for p in priors:
    clf = create_model(features_train, targets_train,p)
    # predict the class labels for the test set
    prediction = clf.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the Naive Bayes Classifier with priors = {p}: {accuracy*100}%")

var_smoothing = [1e-9,1e-8,1e-7,1e-6,1e-5]
for v in var_smoothing:
    clf = create_model(features_train, targets_train,None,v)
    # predict the class labels for the test set
    prediction = clf.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the Naive Bayes Classifier with var_smoothing = {v}: {accuracy*100}%")

