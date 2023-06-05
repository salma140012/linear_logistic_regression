import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Load the dataset


c_data = pd.read_csv(r'C:\Users\User\OneDrive\Documents\car_data.csv')


features = ['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize', 'stroke']
target = 'price'

# Create scatter plots of the selected features against the target variable
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
for i, feature in enumerate(features):
    row = i // 4
    col = i % 4
    axes[row, col].scatter(c_data[feature], c_data[target])
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel(target)
plt.tight_layout()
plt.show()





selected_features = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize']


# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(c_data[selected_features], c_data[target], test_size=0.2)









#Initializing
#initializes a NumPy array theta with all its elements set to zero.
theta = np.zeros(X_train.shape[1])
alpha = 0.0000001
num_iters = 60000

# Implement linear regression with gradient descent
def calc_cost(X, y, theta):
    m = len(y) #number of samples in the dataset
    prediction = X.dot(theta)
    square_err = (prediction - y) ** 2
    J = 1 / (2 * m) * square_err.sum()
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y) #number of samples in the dataset
    cost_history = [] #  store the cost for each iteration in list
    for i in range(num_iters):
        prediction = X.dot(theta)
        error = prediction - y
        theta -= alpha * (1/m) * X.T.dot(error) # made x transpose to be able be multiplied with error
        cost_history.append(calc_cost(X, y, theta))
    return theta, cost_history

theta, cost_history = gradient_descent(X_train, y_train, theta, alpha, num_iters)

# Print the parameters of the hypothesis function
print(theta)

# Plot the cost against the number of iterations
plt.plot(range(num_iters), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Use the optimized hypothesis function to make predictions on the testing set
predictions = X_test.dot(theta)
mse = ((predictions - y_test) ** 2).mean()


# Calculate the accuracy of the final model
accuracy = 1 - mse / np.var(y_test)
print("Accuracy of the final model: ", accuracy)



