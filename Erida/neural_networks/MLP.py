import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.randn(input_size, hidden_size)
    bias_hidden = np.random.randn(hidden_size)
    weights_hidden_output = np.random.randn(hidden_size, output_size)
    bias_output = np.random.randn(output_size)
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

def forward(x, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    hidden_input = np.dot(x, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)
    return hidden_output, final_output

def backward(x, y, hidden_output, final_output, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, learning_rate):
    error = y - final_output
    d_final_output = error * sigmoid_derivative(final_output)

    error_hidden_layer = d_final_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_output)

    weights_hidden_output += hidden_output.T.dot(d_final_output) * learning_rate
    bias_output += np.sum(d_final_output, axis=0) * learning_rate

    weights_input_hidden += x.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0) * learning_rate

    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    input_size = 2
    hidden_size = 2
    output_size = 1
    learning_rate = 0.1
    epochs = 10000

    weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        hidden_output, final_output = forward(X, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
        weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = backward(X, y, hidden_output, final_output, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, learning_rate)

        if epoch % 1000 == 0:
            loss = np.mean(np.square(y - final_output))
            print(f'Эпоха {epoch}, Потери: {loss}')

    hidden_output, final_output = forward(X, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
    print("Предсказание :")
    print(final_output)
