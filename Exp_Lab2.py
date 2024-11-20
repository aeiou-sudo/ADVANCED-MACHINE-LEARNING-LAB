from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def encode_features(data, columns):
    le = LabelEncoder()
    for column in columns:
        data[column] = le.fit_transform(data[column])
    return data

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    data = data.drop(columns=['Margin', 'Match Date', 'T-20 Int Match'])
    data = encode_features(data, ['Team1', 'Team2', 'Ground', 'Winner'])
    features = [col for col in data.columns if col != 'Winner']
    X = data[features]
    y = data['Winner']
    return X, y, data

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def classify_user_input(model, feature_columns, data):
    user_input = []
    le = LabelEncoder()
    label_encodings = {}

    # Fit the label encoder based on the original data's encodings and store valid options
    for column in ['Team1', 'Team2', 'Ground']:
        le.fit(data[column])
        label_encodings[column] = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Valid options for {column}: {list(label_encodings[column].keys())}")

    print("Enter the values for the following features:")
    for feature in feature_columns:
        value = input(f"{feature}: ")
        
        # Encode categorical inputs based on training data's encoding
        if feature in label_encodings:
            if value in label_encodings[feature]:
                encoded_value = label_encodings[feature][value]
                user_input.append(encoded_value)
            else:
                print(f"Invalid input for {feature}. Please enter a known value from the list above.")
                return
        else:
            user_input.append(float(value) if value.replace('.', '', 1).isdigit() else value)

    user_data = pd.DataFrame([user_input], columns=feature_columns)
    prediction = model.predict(user_data)
    print("Predicted Winner:", prediction[0])


X, y, data = preprocess_data('./Dataset/naive_bayesian.csv')
X_train, X_test, y_train, y_test = split_data(X, y)
model = train_model(X_train, y_train)
accuracy = evaluate_model(model, X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Call classify_user_input to classify a new user input based on the trained model
classify_user_input(model, X.columns, data)
