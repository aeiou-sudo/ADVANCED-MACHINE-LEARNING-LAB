# Importing necessary libraries
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
from sklearn.linear_model import LinearRegression  # For performing linear regression
import matplotlib.pyplot as plt #For visual represenation using graphs
import seaborn as sns #For visual represenation using graphs
from mpl_toolkits.mplot3d import Axes3D #For visual represenation using graphs
import numpy as np # For data manipulation

# Loading the dataset from the given file path
file_path = '/Users/pauljose/Downloads/small_student_linear_regression_data.csv'  # Specify the correct file path for the dataset
df = pd.read_csv(file_path)  # Reading the dataset into a pandas DataFrame

# Separating the dataset into features (X) and the target variable (y)
# X contains the independent variables, and y contains the dependent variable (Performance Index)
X = df.drop('Performance Index', axis=1)  # Dropping the 'Performance Index' column from the features
y = df['Performance Index']  # 'Performance Index' is the dependent variable (target)

# Handling categorical data by mapping 'Yes'/'No' in the 'Extracurricular Activities' column to 1/0
# Regression models cannot work with string values, so we convert them to numerical values
X['Extracurricular Activities'] = X['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Splitting the dataset into training and testing sets
# 80% of the data will be used for training, and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# For simple linear regression, we reduce the features to only 'Hours Studied'
# This allows us to predict performance based on just one feature
X_train_simple = X_train[['Hours Studied']]  # Using only 'Hours Studied' for training
# X_train_simple = X_train.iloc[:, [0]]  # Another way to select the first column

# Similarly, reducing the features in the testing set to only 'Hours Studied'
X_test_simple = X_test[['Hours Studied']]

# Creating and training the model for simple linear regression using 'Hours Studied' as the only feature
model_simple = LinearRegression()  # Initializing the linear regression model
model_simple.fit(X_train_simple, y_train)  # Training the model on the training data

# Creating and training the model for multiple linear regression using all features
model_multi = LinearRegression()  # Initializing the multiple linear regression model
model_multi.fit(X_train, y_train)  # Training the model on the full feature set


    # # Prediction for Simple Linear Regression (with feature name for 'Hours Studied')
    # # Wrapping the input in a DataFrame with the same feature name as the training data
    # input_simple = pd.DataFrame({'Hours Studied': [7]})  # Creating a DataFrame with the feature name
    # prediction_simple = model_simple.predict(input_simple)
    # print("Predicted score for 7 hours studied (Simple Linear Regression):", prediction_simple[0])

    # # Prediction for Multiple Linear Regression (with feature names for all features)
    # # Wrapping the input in a DataFrame with the correct feature names
    # input_multi = pd.DataFrame({'Hours Studied': [7], 
    #                             'Previous Scores': [76], 
    #                             'Extracurricular Activities': [1], 
    #                             'Sleep Hours': [15], 
    #                             'Sample Question Papers Practiced': [1]})  # Creating a DataFrame with feature names
    # prediction_multi = model_multi.predict(input_multi)
    # print("Predicted score for 7 hours studied with other factors (Multiple Linear Regression):", prediction_multi[0])

# Make predictions using the Simple Linear Regression model
# The `predict` method takes the feature data for the test set (X_test_simple) 
# and returns the predicted values based on the fitted model.
y_pred_simple = model_simple.predict(X_test_simple)

# Make predictions using the Multiple Linear Regression model
# The `predict` method takes the feature data for the test set (X_test)
# and returns the predicted values based on the fitted model, which includes multiple features.
y_pred_multiple = model_multi.predict(X_test)


    # # Print the actual vs predicted values for the first 5 instances (Simple Linear Regression)
    # print("\nSimple Linear Regression: Actual vs Predicted")
    # for actual, predicted in zip(y_test[:5], y_pred_simple[:5]):
    #     print(f"Actual: {actual}, Predicted: {predicted}")

    # # Print the actual vs predicted values for the first 5 instances (Multiple Linear Regression)
    # print("\nMultiple Linear Regression: Actual vs Predicted")
    # for actual, predicted in zip(y_test[:5], y_pred_multiple[:5]):
    #     print(f"Actual: {actual}, Predicted: {predicted}")

# Calculate Mean Squared Error for Simple Linear Regression
# mse_simple = (1/n) * Σ(actual - predicted)²
mse_simple = sum((y_test - y_pred_simple) ** 2) / len(y_test)

# Calculate Mean Squared Error for Multiple Linear Regression
# mse_multiple = (1/n) * Σ(actual - predicted)²
mse_multiple = sum((y_test - y_pred_multiple) ** 2) / len(y_test)

# Print the MSE results for both models
print("\nMean Squared Error for Simple Linear Regression (mse_simple):", mse_simple)
print("\nMean Squared Error for Multiple Linear Regression (mse_multiple):", mse_multiple)
print("\n")

def plotGraph_SLR():
    # Plotting Simple Linear Regression
    plt.figure(figsize=(10, 6))

    # Scatter plot of actual values
    plt.scatter(X_test_simple, y_test, color='blue', label='Actual Values')

    # Line plot for predicted values
    plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Predicted Values')

    # Adding titles and labels
    plt.title('Simple Linear Regression: Actual vs Predicted')
    plt.xlabel('Hours Studied')
    plt.ylabel('Performance Index')
    plt.legend()

    # Show the plot
    plt.show()

def plotGraph_MLR():
    # Create a 3D plot for Multiple Linear Regression
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of actual values
    ax.scatter(X_test['Hours Studied'], X_test['Previous Scores'], y_test, color='blue', label='Actual Values')

    # Create a grid to plot the predictions
    x1_range = np.linspace(X_test['Hours Studied'].min(), X_test['Hours Studied'].max(), 10)
    x2_range = np.linspace(X_test['Previous Scores'].min(), X_test['Previous Scores'].max(), 10)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    # Use the mean or median values for the other features
    mean_extracurricular = X_train['Extracurricular Activities'].mean()
    mean_sleep_hours = X_train['Sleep Hours'].mean()
    mean_sample_questions = X_train['Sample Question Papers Practiced'].mean()

    # Generate predictions for the grid, including all features
    input_grid = pd.DataFrame({
        'Hours Studied': x1_grid.ravel(),
        'Previous Scores': x2_grid.ravel(),
        'Extracurricular Activities': np.full_like(x1_grid.ravel(), mean_extracurricular),
        'Sleep Hours': np.full_like(x1_grid.ravel(), mean_sleep_hours),
        'Sample Question Papers Practiced': np.full_like(x1_grid.ravel(), mean_sample_questions)
    })

    y_pred_grid = model_multi.predict(input_grid)
    y_pred_grid = y_pred_grid.reshape(x1_grid.shape)

    # Plot the surface
    ax.plot_surface(x1_grid, x2_grid, y_pred_grid, color='red', alpha=0.5)

    # Adding titles and labels
    ax.set_title('Multiple Linear Regression: Actual vs Predicted')
    ax.set_xlabel('Hours Studied')
    ax.set_ylabel('Previous Scores')
    ax.set_zlabel('Performance Index')

    # Show the plot
    plt.legend()
    plt.show()

plotGraph_SLR()
plotGraph_MLR()
    