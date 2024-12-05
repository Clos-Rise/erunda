# тут половину спиздил кстати

import numpy as np

class PenisMachine:
    def __init__(self, num_visible, num_hidden, learning_rate=0.1):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.weights = np.random.randn(num_visible, num_hidden) * 0.1
        self.hidden_bias = np.zeros(num_hidden)
        self.visible_bias = np.zeros(num_visible)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def test_hidden(self, visible):
        activation = np.dot(visible, self.weights) + self.hidden_bias
        prob = self.sigmoid(activation)
        return (prob > np.random.rand(self.num_hidden)).astype(int)

    def test_visible(self, hidden):
        activation = np.dot(hidden, self.weights.T) + self.visible_bias
        prob = self.sigmoid(activation)
        return (prob > np.random.rand(self.num_visible)).astype(int)

    def train(self, data, epochs=100):
        for epoch in range(epochs):
            for visible in data:
                hidden = self.sample_hidden(visible)
                visible_reconstruction = self.sample_visible(hidden)
                hidden_reconstruction = self.sample_hidden(visible_reconstruction)

                positive_grad = np.outer(visible, hidden)
                negative_grad = np.outer(visible_reconstruction, hidden_reconstruction)

                self.weights += self.learning_rate * (positive_grad - negative_grad)
                self.visible_bias += self.learning_rate * (visible - visible_reconstruction)
                self.hidden_bias += self.learning_rate * (hidden - hidden_reconstruction)

    def generate(self, num_samples):
        samples = np.zeros((num_samples, self.num_visible), dtype=int)
        for i in range(num_samples):
            visible = np.random.randint(0, 2, self.num_visible)
            for _ in range(100):
                hidden = self.test_hidden(visible)
                visible = self.test_visible(hidden)
            samples[i] = visible
        return samples

if __name__ == "__main__":
    np.random.seed(42)
    data = np.array([
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])

    bm = PenisMachine(num_visible=4, num_hidden=2, learning_rate=0.1)
    bm.train(data, epochs=1000)

    generated_samples = bm.generate(num_samples=5)
    print("Сгенерированный бред:")
    print(generated_samples)
