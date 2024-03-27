import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((output_size, 1))
        self.activation_function = activation_function
        self.output = None
        self.input = None
        self.weight_gradient = None
        self.bias_gradient = None

    def forward(self, input_data):
        """Compute the output of this layer using the given input."""
        self.input = input_data
        z = np.dot(self.weights, input_data) + self.biases
        self.output = self.activation_function.function(z)
        return self.output

    def backward(self, output_gradient, learning_rate):
        """Backpropagate the gradient through this layer."""
        # Compute derivative of the activation function
        activation_derivative = self.activation_function.derivative(self.output)
        
        # Apply chain rule (element-wise multiplication with activation derivative)
        delta = output_gradient * activation_derivative
        
        # Compute gradients w.r.t weights and biases
        self.weight_gradient = np.dot(delta, self.input.T)
        self.bias_gradient = np.sum(delta, axis=1, keepdims=True)
        
        # Update weights and biases
        self.weights -= learning_rate * self.weight_gradient
        self.biases -= learning_rate * self.bias_gradient
        
        # Compute and return the gradient for the input layer (for further backpropagation)
        return np.dot(self.weights.T, delta)