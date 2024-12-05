import numpy as np

class PizdaNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        self.weights = np.zeros((self.size, self.size))
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def predict(self, pattern, steps=10):
        pattern = np.array(pattern)
        for _ in range(steps):
            for i in range(self.size):
                pattern[i] = np.sign(np.dot(self.weights[i], pattern))
        return pattern

if __name__ == "__main__":
    patterns = [
        np.array([1, -1, 1, -1]),
        np.array([-1, 1, -1, 1]),
        np.array([1, 1, -1, -1])
    ]

    pizdec_net = PizdaNetwork(size=4)
    pizdec_net.train(patterns)

    noisy_pattern = np.array([-1, -1, -1, -1])
    recalled_pattern = pizdec_net.predict(noisy_pattern)

    print("Исходный:", noisy_pattern)
    print("Восстановлен:", recalled_pattern)
