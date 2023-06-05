import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load the dataset
customer_data = pd.read_csv(r'C:\Users\User\OneDrive\Documents\customer_data.csv')


# Normalize the age and salary columns
customer_data['age'] = (customer_data['age'] - customer_data['age'].min()) / (customer_data['age'].max() - customer_data['age'].min())
customer_data['salary'] = (customer_data['salary'] - customer_data['salary'].min()) / (customer_data['salary'].max() - customer_data['salary'].min())


# Separate the data into the two classes
purchased_0 = customer_data[customer_data['purchased'] == 0]
purchased_1 = customer_data[customer_data['purchased'] == 1]

# Create scatter plots for each class
plt.scatter(purchased_0['age'], purchased_0['salary'], color='red', label='Not Purchased')
plt.scatter(purchased_1['age'], purchased_1['salary'], color='blue', label='Purchased')

# Add axis labels and legend
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()

# Show the plot
plt.show()


# Split the dataset into training and testing sets (80-20 split)
train_size = int(0.8 * len(customer_data))

X_train = customer_data[['age', 'salary']][:train_size].values
y_train = customer_data['purchased'][:train_size].values

X_test = customer_data[['age', 'salary']][train_size:].values
y_test = customer_data['purchased'][train_size:].values

# Define the model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Print the parameters of the hypothesis function
print('Coefficients(theta values): ', model.coef_)
print('Intercept: ', model.intercept_)

# Use the optimized hypothesis function to make predictions on the testing set
predictions = model.predict(X_test)

# Calculate the accuracy of the final model
accuracy = model.score(X_test, y_test)
print('Accuracy of the final model: ', accuracy)
