import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model_simple():
    # Load the dataset
    df = pd.read_csv('./Dataset/simpleLR.csv')
    
    # Separate features (X) and target (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Display the model's coefficients and intercept
    print('Model parameters:')
    print(f'Coefficients: {model.coef_}')
    print(f'Intercept: {model.intercept_}\n')

    # Calculate Mean Squared Error (MSE)
    y_pred = model.predict(X_test)
    mean_square_error = np.mean((y_test - y_pred) ** 2)
    print(f'Mean Squared Error: {mean_square_error}\n')

    def get_user_input():
        # Get user input
        user_input = float(input(f'Enter the value for {X.columns[0]}: '))
        
        # Create a DataFrame with the correct feature name
        input_df = pd.DataFrame([[user_input]], columns=[X.columns[0]])
        
        # Make the prediction
        result = model.predict(input_df)
        print(f'Predicted value: {result[0]}')

    # Call the function
    get_user_input()

def train_model_multiple():
    # Load the dataset
    df = pd.read_csv('./Dataset/multipleLR.csv')
    
    # Separate features (X) and target (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    

def main():
    train_model_simple()

if __name__ == "__main__":
    main()
