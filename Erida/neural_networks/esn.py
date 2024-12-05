import numpy as np

class ESN:
    def __init__(self, n_inputs, n_outputs, n_reservoir=100, spectral_radius=0.9, sparsity=0.1, input_scaling=1.0, ridge_reg=1e-8):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.ridge_reg = ridge_reg

        self.W_in = np.random.rand(n_reservoir, n_inputs) * 2 - 1
        self.W = np.random.rand(n_reservoir, n_reservoir) * 2 - 1
        self.W_out = None

        self.W *= spectral_radius / max(abs(np.linalg.eigvals(self.W)))
        self.W *= np.random.rand(n_reservoir, n_reservoir) < sparsity

    def fit(self, X, y):
        n_samples = X.shape[0]
        X_ext = np.zeros((n_samples, self.n_reservoir))
        state = np.zeros(self.n_reservoir)

        for t in range(n_samples):
            state = np.tanh(np.dot(self.W_in, X[t]) + np.dot(self.W, state))
            X_ext[t] = state
        X_train = np.hstack([X, X_ext])

        self.W_out = np.linalg.inv(np.dot(X_train.T, X_train) + self.ridge_reg * np.eye(X_train.shape[1]))
        self.W_out = np.dot(self.W_out, np.dot(X_train.T, y))

    def predict(self, X):
        n_samples = X.shape[0]
        X_ext = np.zeros((n_samples, self.n_reservoir))
        state = np.zeros(self.n_reservoir)

        for t in range(n_samples):
            state = np.tanh(np.dot(self.W_in, X[t]) + np.dot(self.W, state))
            X_ext[t] = state

        X_pred = np.hstack([X, X_ext])
        return np.dot(X_pred, self.W_out)
if __name__ == "__main__":
    np.random.seed(42)
    X_train = np.random.rand(100, 1)
    y_train = np.sin(X_train)

    esn = ESN(n_inputs=1, n_outputs=1, n_reservoir=100)
    esn.fit(X_train, y_train)

    X_test = np.random.rand(10, 1)
    y_test = np.sin(X_test)

    y_pred = esn.predict(X_test)
    print("Исходные:", y_test.flatten())
    print("Предсказ:", y_pred.flatten())
