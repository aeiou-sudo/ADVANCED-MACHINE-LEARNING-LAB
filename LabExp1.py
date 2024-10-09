# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
    X_train_simple = X_train.iloc[:, [0]]  # Use only the first feature
    X_test_simple = X_test.iloc[:, [0]]
    
    lr.fit(X_train_simple, y_train)
    y_pred = lr.predict(X_test_simple)
    
    # mse = mean_squared_error(y_test, y_pred)
    
    print("\nSimple Linear Regression:")
    print("Coefficients:", lr.coef_)
    print("Intercept:", lr.intercept_)
    # print("Mean Squared Error:", mse)

# Step 4: Multiple Linear Regression (Multiple features)
def multiple_linear_regression(X_train, y_train, X_test, y_test):
    lr = LinearRegression()
    
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    # mse = mean_squared_error(y_test, y_pred)
    
    print("\nMultiple Linear Regression:")
    print("Coefficients:", lr.coef_)
    print("Intercept:", lr.intercept_)
    # print("Mean Squared Error:", mse)

# Step 5: Main function to execute the regression process
def main():
    # Load the dataset from CSV
    file_path = './Linear_Reg_DATASET.csv'  # Replace with the actual file path
    data = load_data(file_path)
    
    # Preprocess the data (split into X and y)
    target_column = 'Salary'  # Replace with the actual target column name
    X, y = preprocess_data(data, target_column)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Simple Linear Regression
    simple_linear_regression(X_train, y_train, X_test, y_test)
    
    # Step 4: Multiple Linear Regression
    multiple_linear_regression(X_train, y_train, X_test, y_test)

# Run the main function
if __name__ == "__main__":
    main()
