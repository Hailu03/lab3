from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# load dataset
diabetes = load_diabetes()

def create_model(features_train, targets_train):
    # create model
    model = LinearRegression()
    model.fit(features_train, targets_train)
    return model

def predict(model, features_test):
    pred = model.predict(features_test)
    return pred

# divide dataset into features and targets
features = diabetes.data
targets = diabetes.target

# shuffle the dataset
randNum = np.arange(features.shape[0])
np.random.shuffle(randNum)
features = features[randNum]
targets = targets[randNum]

# get the feature names
feature_names = diabetes.feature_names
colors = ['red', 'green', 'blue', 'red', 'orange', 'purple', 'pink', 'brown', 'gray', 'violet']

# training dataset in each feature
plt.figure(figsize=(16, 10)) # set the size of the figure
for i in range(0, 10):
    feature = features[:, i] # get the feature

    # divide dataset into training and testing sets
    feature_train, feature_test, target_train, target_test = train_test_split(feature, targets, test_size=0.2)
    
    # reshape the data (1D -> 2D) (353,) -> (353,1) to fit the model
    feature_train = feature_train.reshape(-1, 1)
    feature_test = feature_test.reshape(-1, 1)

    # create model
    model = create_model(feature_train, target_train)

    # predict
    pred = predict(model, feature_test)

    # calculate mean squared error
    print(f"Error of the Linear Regresson for feature {i+1}: ", mean_squared_error(target_test, pred))

    # plot the data
    plt.subplot(2,5,i+1)
    plt.title(f"Feature {i+1}: {feature_names[i]}") # set the title of the figure
    plt.scatter(feature_test, target_test, color=colors[i], label='Data') # plot the data
    plt.xlabel('Feature') # set the label of the x-axis 
    plt.ylabel('Target') # set the label of the y-axis
    plt.plot(feature_test, pred, color='black', linewidth=3) # plot the line
    plt.tight_layout()

plt.show() # show the figure

