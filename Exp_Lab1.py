import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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

def multiple_LR():
    print("Running Multiple Linear Regression")
    df = readFile_asDF('multipleLR')
    X, y = separateDataset(df)
    
# Strip any extra whitespace from State values, then encode the categorical data
    if 'State' in X.columns:
        X['State'] = X['State'].str.strip()  # Remove extra whitespaces
        X = pd.get_dummies(X, columns=['State'], drop_first=True)  # One-hot encode the 'State' column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_multi = LinearRegression()
    model_multi.fit(X_train, y_train)
    y_pred = model_predict(model_multi, X_test)
    MeanSquaredError(y_pred, y_test)

def main():
    simple_LR()
    multiple_LR()

if __name__ == "__main__":
    main()