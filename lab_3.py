#this is the new file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
np.random.seed(2)
# load the Iris dataset
df_1 = pd.read_csv('D:/Year_2/Source_code/DataScience/Data_sets/Iris.csv')
print(df_1[:20])
# replace the categorical species column with numerical values
df_1['Species'].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], inplace=True)
# handle missing values 5% Nan
nan_mask = np.random.choice([False, True], size=df_1.shape, p=[0.95, 0.05])
df_1[nan_mask] = np.nan
df_1 = df_1.mean()
# split the data into training and testing sets
X = df_1.drop('Species', axis=1)
y = df_1['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=125)
# Build a Gaussian Classifier
nb = GaussianNB()
# Model training
nb.fit(X_train, y_train)
# Predict Output
y_pred = nb.predict(X_test)
# evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# print the data
