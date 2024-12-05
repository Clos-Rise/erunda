import numpy as np

def activ(x):
    return 1 if x >= 0 else 0

def perceptronus(x, w, b):
    return [activ(np.dot(w[i], x) + b[i]) for i in range(len(w))]

x = np.array([1, 2, 3])
w = np.array([
    [0.5, -0.5, 0.2],
    [0.3, 0.1, -0.4],
    [-0.2, 0.6, 0.3]
])
b = np.array([0.1, -0.2, 0.3]) # <- эт смещения если че

output = perceptronus(x, w, b)
print("Вывод:", output)
