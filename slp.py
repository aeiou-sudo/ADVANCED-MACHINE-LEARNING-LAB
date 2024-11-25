import numpy as np

# Activation function: Step function
def step_function(x):
    return 1 if x >= 0 else 0

# Perceptron class
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)  # Initialize weights to zero
        self.bias = 0  # Initialize bias to zero
    
    # Predict output for a given input
    def predict(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        return step_function(weighted_sum)
    
    # Train the perceptron on a dataset
    def train(self, X, y, epochs=10, learning_rate=0.1):
        for epoch in range(epochs):
            for i in range(len(X)):
                # Prediction
                prediction = self.predict(X[i])
                # Update the weights and bias if there is an error
                error = y[i] - prediction
                self.weights += learning_rate * error * X[i]
                self.bias += learning_rate * error
            print(f'Epoch {epoch+1}: Weights = {self.weights}, Bias = {self.bias}')
                
# Define the logic gate data
# Input data for AND, OR, XOR gates (2 input bits)
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])  # AND gate output

X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])   # OR gate output

# Initialize perceptron for AND gate
perceptron_and = Perceptron(input_size=2)
print("Training Perceptron for AND Gate:")
perceptron_and.train(X_and, y_and, epochs=10)

# Initialize perceptron for OR gate
perceptron_or = Perceptron(input_size=2)
print("\nTraining Perceptron for OR Gate:")
perceptron_or.train(X_or, y_or, epochs=10)

# Test the trained perceptrons
print("\nTesting on AND gate data:")
for x in X_and:
    print(f"Input: {x} => Prediction: {perceptron_and.predict(x)}")

print("\nTesting on OR gate data:")
for x in X_or:
    print(f"Input: {x} => Prediction: {perceptron_or.predict(x)}")
