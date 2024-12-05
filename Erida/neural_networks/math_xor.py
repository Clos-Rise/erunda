#СУКА ЗАЛУПА ЕБАНАЯ НИХУЯ НЕ РАБОТАЕЕЕТ!!!

import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralKiev:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)

    def forward(self, inputs):
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden)
        self.hidden_layer_output = relu(self.hidden_layer_input)
        self.final_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.final_layer_output = sigmoid(self.final_layer_input)
        return self.final_layer_output

    def backward(self, inputs, outputs, targets, learning_rate):
        output_error = targets - outputs
        output_delta = output_error * sigmoid_derivative(self.final_layer_output)
        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * relu_derivative(self.hidden_layer_output)
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_layer_delta) * learning_rate

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            outputs = self.forward(inputs)
            self.backward(inputs, outputs, targets, learning_rate)

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralKiev(2, 4, 1)
    nn.train(X, y, epochs=10000, learning_rate=0.1)
    print(nn.forward(X))
