import numpy as np
import neuroplain as nn

# Define the XOR input and output
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0, 1, 1, 0]])

# Initialize the network
network = nn.FFNetwork()
network.add_layer(nn.Layer(2, 16, nn.ActivationFunction.ReLU()))
network.add_layer(nn.Layer(16, 1, nn.ActivationFunction.Sigmoid()))
network.set_loss_function(nn.LossFunction.MSE())

# Train the network
network.train(X_train, y_train.T, epochs=1000, learning_rate=0.01)

# Test the network
print("Testing trained network on XOR problem:")
for x, y in zip(X_train, y_train.T):
    predicted = network.predict(x)
    print(f"Input: {x} - True: {y} - Predicted: {predicted.ravel()}")
