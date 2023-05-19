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
y = iris.target  

# shuffle the dataset
randNum = np.arange(X.shape[0])
np.random.shuffle(randNum)
X = X[randNum]
y = y[randNum]

# choose two classes randomly
class1_label = np.random.randint(0, 3)
while True:
    class2_label = np.random.randint(0, 3)
    if class2_label != class1_label:
        break

class1_samples = X[y == class1_label]
class2_samples = X[y == class2_label]

# Combine the samples from the two classes
filtered_samples = np.concatenate((class1_samples, class2_samples), axis=0)
filtered_labels = np.concatenate((np.zeros(len(class1_samples)), np.ones(len(class2_samples))), axis=0)

# divide dataset into training and testing sets with a ratio of 80/20
features_train, features_test, targets_train, targets_test = train_test_split(filtered_samples, filtered_labels, test_size=0.2)

# create logistic regression model
model = LogisticRegression()
model.fit(features_train, targets_train)

# predict the class labels for the test set
prediction = model.predict(features_test)

# calculate the accuracy of the trained model on the testing set
accuracy = accuracy_score(targets_test, prediction)
print(f"Accuracy of the Logistic Regression: {accuracy*100}%")
