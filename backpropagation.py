import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Sigmoid Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)  # Input to hidden weights
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)  # Hidden to output weights
        self.bias_hidden = np.zeros((1, self.hidden_size))  # Bias for hidden layer
        self.bias_output = np.zeros((1, self.output_size))  # Bias for output layer

    # Forward pass
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output
    
    # Backpropagation
    def backward(self, X, y, learning_rate):
        output_error = y - self.final_output  # Error at the output
        output_delta = output_error * sigmoid_derivative(self.final_output)  # Delta at the output layer
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)  # Error at the hidden layer
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)  # Delta at the hidden layer
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # Train the model
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))  # Mean Squared Error Loss
                print(f"Epoch {epoch}/{epochs}, Loss: {loss}")
                
    # Predict the output
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)  # Return the index of the max output class

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# One-hot encoding of the target labels
y_encoded = np.zeros((y.size, 3))
y_encoded[np.arange(y.size), y] = 1

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Initialize the Neural Network
input_size = X_train.shape[1]  # 4 input features (sepal length, sepal width, petal length, petal width)
hidden_size = 5  # Hidden layer size (adjustable)
output_size = y_train.shape[1]  # 3 classes of output (Setosa, Versicolor, Virginica)
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the Neural Network
nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)

# Test the Neural Network
y_pred = nn.predict(X_test)

# Convert one-hot encoded test labels back to single class labels
y_test_labels = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test_labels, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

