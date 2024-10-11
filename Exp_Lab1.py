# Importing necessary libraries
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
from sklearn.linear_model import LinearRegression  # For performing linear regression

# Loading the dataset from the given file path
file_path = '/Users/pauljose/Downloads/student_linear_regression_data.csv'  # Specify the correct file path for the dataset
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


# Prediction for Simple Linear Regression (with feature name for 'Hours Studied')
# Wrapping the input in a DataFrame with the same feature name as the training data
input_simple = pd.DataFrame({'Hours Studied': [7]})  # Creating a DataFrame with the feature name
prediction_simple = model_simple.predict(input_simple)
print("Predicted score for 7 hours studied (Simple Linear Regression):", prediction_simple[0])

# Prediction for Multiple Linear Regression (with feature names for all features)
# Wrapping the input in a DataFrame with the correct feature names
input_multi = pd.DataFrame({'Hours Studied': [7], 
                            'Previous Scores': [99], 
                            'Extracurricular Activities': [1], 
                            'Sleep Hours': [9], 
                            'Sample Question Papers Practiced': [1]})  # Creating a DataFrame with feature names
prediction_multi = model_multi.predict(input_multi)
print("Predicted score for 7 hours studied with other factors (Multiple Linear Regression):", prediction_multi[0])
