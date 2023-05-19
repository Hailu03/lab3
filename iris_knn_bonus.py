import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_model(features_train, targets_train,n_neighbors,metric='minkowski',weight='uniform',algorithm='auto',leaf_size=30,p=2):
    # create model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,metric=metric,weights=weight,algorithm=algorithm,leaf_size=leaf_size,p=p)

    # Train the model using the training set
    knn.fit(features_train, targets_train)
    return knn

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
n_neighbors = [2,3,4,5,6,7,8,9,10]
for n in n_neighbors:
    knn = create_model(features_train, targets_train,n)
    # predict the class labels for the test set
    prediction = knn.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the KNN Classifier with n_neighbor = {n}: {accuracy*100}%")

print()
metrics = ['euclidean','manhattan','chebyshev','minkowski']
for m in metrics:
    knn = create_model(features_train, targets_train,3,m)
    # predict the class labels for the test set
    prediction = knn.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the KNN Classifier with metric = {m}: {accuracy*100}%")

print()
weights = ['uniform','distance']
for w in weights:
    knn = create_model(features_train, targets_train,3,'minkowski',w)
    # predict the class labels for the test set
    prediction = knn.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the KNN Classifier with weights = {w}: {accuracy*100}%")

print()
algorithm = ['auto','ball_tree','kd_tree','brute']
for a in algorithm:
    knn = create_model(features_train, targets_train,3,'minkowski','uniform',a)
    # predict the class labels for the test set
    prediction = knn.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the KNN Classifier with algorithm = {a}: {accuracy*100}%")

print()
leaf_size = [10,20,30,40,50]
for l in leaf_size:
    knn = create_model(features_train, targets_train,3,'minkowski','uniform','auto',l)
    # predict the class labels for the test set
    prediction = knn.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the KNN Classifier with leaf_size = {l}: {accuracy*100}%")

print()
p = [1,2,3,4,5]
for p_value in p:
    knn = create_model(features_train, targets_train,3,'minkowski','uniform','auto',30,p_value)
    # predict the class labels for the test set
    prediction = knn.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the KNN Classifier with p = {p_value}: {accuracy*100}%")