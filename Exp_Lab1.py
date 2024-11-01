import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import time

def readFile_asDF(dataset):
    file_path = f'./Dataset/{dataset}.csv'
    return pd.read_csv(file_path)

def separateDataset(df):
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    return X, y

def model_predict(model, X):
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print("Model is predicting...")
    time.sleep(2)
    return model.predict(X)

def MeanSquaredError(y_obtained, y_target):
    mse = np.mean((y_target - y_obtained) ** 2)
    print(f'Mean squared error: {mse}')

def simple_LR():
    print("Running Simple Linear Regression")
    df = readFile_asDF('simpleLR')
    X, y = separateDataset(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_simple = LinearRegression()
    model_simple.fit(X_train, y_train)
    y_pred = model_predict(model_simple, X_test)
    MeanSquaredError(y_pred, y_test)
    return model_simple

def multiple_LR():
    print("Running Multiple Linear Regression")
    df = readFile_asDF('multipleLR')
    X, y = separateDataset(df)
    if 'State' in X.columns:
        X['State'] = X['State'].str.strip()  # Remove extra whitespaces
        X = pd.get_dummies(X, columns=['State'], drop_first=True)  # One-hot encode the 'State' column
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_multi = LinearRegression()
    model_multi.fit(X_train, y_train)
    y_pred = model_predict(model_multi, X_test)
    MeanSquaredError(y_pred, y_test)
    return model_multi

def main():
    print("Running Simple Linear Regression")
    model_simple = simple_LR()
    print('<--------------------------->')
    
    print("Running Multiple Linear Regression")
    model_multi = multiple_LR()
    print('<--------------------------->')
    
    userInput = 1
    while userInput:
        userInput = int(input("Choose from the menu below:\n1. Predict Simple Linear Regression\n2. Predict Multiple Linear Regression\n0. Exit\n: "))
        
        if userInput == 1:
            X = float(input("Enter a value for X: "))
            X = np.array([[X]])
            y_pred = model_predict(model_simple, X)
            print("Predicted value for y:", y_pred[0])
        
        elif userInput == 2:
            X_values = input("Enter values for X (comma-separated, e.g., 5,3,2,4,1): ")
            # Map categorical variables before creating the input array
            X_list = X_values.split(",")
            X_numeric = []
            
            for value in X_list:
                value = value.strip()
                if value in ['True', 'False']:
                    X_numeric.append(int(value == 'True'))  # Convert boolean strings to integers
                else:
                    X_numeric.append(float(value))  # Convert other numeric inputs
                    
            X = np.array(X_numeric).reshape(1, -1)
            y_pred = model_predict(model_multi, X)
            print("Predicted value for y:", y_pred[0])
        elif userInput == 0:
            print("Exiting the program.")
        
        else:
            print(f'Invalid option: {userInput}.')

if __name__ == "__main__":
    main()
