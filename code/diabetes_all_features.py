from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# load dataset
diabetes = load_diabetes()

# describe dataset
print(diabetes.DESCR)

# divide dataset into features and targets
features = diabetes.data
targets = diabetes.target

# shuffle the dataset
randNum = np.arange(features.shape[0])
np.random.shuffle(randNum)
features = features[randNum]
targets = targets[randNum]

# divide dataset into training and testing sets
features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.2)

# create model
model = LinearRegression()
model.fit(features_train, targets_train)
pred = model.predict(features_test)

# calculate mean squared error
print("Error of the Linear Regresson: ", mean_squared_error(targets_test, pred))
