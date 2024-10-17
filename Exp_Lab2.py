# Importing necessary libraries
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
from sklearn.naive_bayes import GaussianNB  # For performing linear regression
import matplotlib.pyplot as plt #For visual represenation using graphs
import seaborn as sns
import numpy as np # For data manipulation
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# Load the dataset from a CSV file
# Replace 'sample_data.csv' with your actual dataset file
data = pd.read_csv('./Dataset/sample_data.csv')

# Print the dataset for inspection
print(data.head())


# Initialize the LabelEncoder
le = LabelEncoder()

# Convert categorical features into numerical values
for column in ['Attends Workshop', 'Has Arts Background', 'Parent Encouragement', 'School Support', 'Has Free Time', 'Participates']:
    data[column] = le.fit_transform(data[column])


# Print the dataset for inspection
print(data.head())

# Assume the last column is the target (label) and others are features
X = data.iloc[:, :-1]  # All columns except the last
y = data.iloc[:, -1]   # The last column is the target

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
model = GaussianNB()

# Train the classifier on the training set
model.fit(X_train, y_train)

# Predict the target values for the test set
y_pred = model.predict(X_test)

# Compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Naive Bayes Classifier: {accuracy * 100:.2f}%")

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Get predicted probabilities for the positive class (1)
y_prob = model.predict_proba(X_test)[:, 1]

# Compute the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()