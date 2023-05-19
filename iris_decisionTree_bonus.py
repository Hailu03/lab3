import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_model(features_train, targets_train,criterion='gini',splitter='best',max_depth=None,min_samples_split=2,
                 min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_features=None,max_leaf_nodes=None,
                 min_impurity_decrease=0.0,ccp_alpha=0.0,class_weight=None):
    
    # create model
    knn = DecisionTreeClassifier(criterion = criterion,splitter=splitter,max_depth=max_depth,
                                min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,max_features=max_features,
                                max_leaf_nodes=max_leaf_nodes,min_impurity_decrease=min_impurity_decrease,
                                ccp_alpha=ccp_alpha,class_weight=class_weight
                                )

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

# create model with differnet hyperparameters
criteria = ['gini','entropy','log_loss']
for c in criteria:
    clf = create_model(features_train, targets_train,criterion=c)
    # predict the class labels for the test set
    prediction = clf.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the Decision Tree Classifier with criteria = {c}: {accuracy*100}%")

splitter = ['best','random']
for s in splitter:
    clf = create_model(features_train, targets_train,splitter=s)
    # predict the class labels for the test set
    prediction = clf.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the Decision Tree Classifier with splitter = {s}: {accuracy*100}%")

max_depth = [None,1,2,3,4,5,6,7,8,9,10]
for m in max_depth:
    clf = create_model(features_train, targets_train,max_depth=m)
    # predict the class labels for the test set
    prediction = clf.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the Decision Tree Classifier with max_depth = {m}: {accuracy*100}%")

min_samples_split = [2,3,4,5,6,7,8,9,10]
for m in min_samples_split:
    clf = create_model(features_train, targets_train,min_samples_split=m)
    # predict the class labels for the test set
    prediction = clf.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the Decision Tree Classifier with min_samples_split = {m}: {accuracy*100}%")

min_samples_leaf = [1,2,3,4,5,6,7,8,9,10]
for m in min_samples_leaf:
    clf = create_model(features_train, targets_train,min_samples_leaf=m)
    # predict the class labels for the test set
    prediction = clf.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the Decision Tree Classifier with min_samples_leaf = {m}: {accuracy*100}%")

min_weight_fraction_leaf = [0.0,0.1,0.2,0.3,0.4,0.5]
for m in min_weight_fraction_leaf:
    clf = create_model(features_train, targets_train,min_weight_fraction_leaf=m)
    # predict the class labels for the test set
    prediction = clf.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the Decision Tree Classifier with min_weight_fraction_leaf = {m}: {accuracy*100}%")
max_features = [None,'sqrt','log2']
for m in max_features:
    clf = create_model(features_train, targets_train,max_features=m)
    # predict the class labels for the test set
    prediction = clf.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the Decision Tree Classifier with max_features = {m}: {accuracy*100}%")

max_leaf_nodes = [2,3,4,5,6]
for m in max_leaf_nodes:
    clf = create_model(features_train, targets_train,max_leaf_nodes=m)
    # predict the class labels for the test set
    prediction = clf.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the Decision Tree Classifier with max_leaf_nodes = {m}: {accuracy*100}%")

min_impurity_decrease = [0.0,0.1,0.2,0.3,0.4,0.5]
for m in min_impurity_decrease:
    clf = create_model(features_train, targets_train,min_impurity_decrease=m)
    # predict the class labels for the test set
    prediction = clf.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction)
    print(f"Accuracy of the Decision Tree Classifier with min_impurity_decrease = {m}: {accuracy*100}%")

ccp_alpha = [0.0,0.1,0.2,0.3,0.4,0.5]
for m in ccp_alpha:
    clf = create_model(features_train, targets_train,ccp_alpha=m)
    # predict the class labels for the test set
    prediction = clf.predict(features_test)
    # calculate the accuracy of the trained model on the testing set
    accuracy = accuracy_score(targets_test, prediction) 
    print(f"Accuracy of the Decision Tree Classifier with ccp_alpha = {m}: {accuracy*100}%")


