# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.metrics import mean_squared_error

# Step 1: Load the CSV data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Step 2: Split data into features (X) and target (y)
def preprocess_data(data, target_column):
    X = data.drop(target_column, axis=1)  # Independent variables
    y = data[target_column]               # Dependent variable (target)
    return X, y

# Step 3: Simple Linear Regression (Single feature)
def simple_linear_regression(X_train, y_train, X_test, y_test):
    lr = LinearRegression()
    X_train_simple = X_train.iloc[:, [1]]  # Use only the first feature
    X_test_simple = X_test.iloc[:, [1]]
    
    lr.fit(X_train_simple, y_train)
    y_pred = lr.predict(X_test_simple)
    
    # mse = mean_squared_error(y_test, y_pred)
    
    print("\nSimple Linear Regression:")
    print("Coefficients:", lr.coef_)
    print("Intercept:", lr.intercept_)
    # print("Mean Squared Error:", mse)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_simple, y_test, color='blue', label='Actual')
    plt.scatter(X_test_simple, y_pred, color='red', label='Predicted')
    plt.plot(X_test_simple, y_pred, color='green', linewidth=2, label='Regression Line')
    plt.title('Simple Linear Regression')
    plt.xlabel('Independent Variable (Feature)')
    plt.ylabel('Dependent Variable (Target)')
    plt.legend()
    plt.show()

# Step 4: Multiple Linear Regression (Multiple features)
def multiple_linear_regression(X_train, y_train, X_test, y_test, y):
    lr = LinearRegression()
    
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    # mse = mean_squared_error(y_test, y_pred)
    
    print("\nMultiple Linear Regression:")
    print("Coefficients:", lr.coef_)
    print("Intercept:", lr.intercept_)
    # print("Mean Squared Error:", mse)

    # Plotting the results (for visualization, we will plot actual vs predicted)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')  # Diagonal line
    plt.title('Multiple Linear Regression: Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

# Step 5: Main function to execute the regression process
def main():
    # Load the dataset from CSV
    file_path = '/Users/pauljose/Downloads/multiLinearRawData.csv'  # Replace with the actual file path
    data = load_data(file_path)
    
    # Preprocess the data (split into X and y)
    target_column = 'target'  # Replace with the actual target column name
    X, y = preprocess_data(data, target_column)
    print("X is :\n", X)
    print("Y is: \n", y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Simple Linear Regression
    simple_linear_regression(X_train, y_train, X_test, y_test)
    
    # Step 4: Multiple Linear Regression
    multiple_linear_regression(X_train, y_train, X_test, y_test, y)

# Run the main function
if __name__ == "__main__":
    main()
