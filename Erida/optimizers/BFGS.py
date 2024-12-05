import numpy as np
from scipy.optimize import minimize

def bumbum_function(x):
    return x[0]**2 + x[1]**2 + x[2]**2

initial_guess = np.array([1.0, 1.0, 1.0])
result = minimize(bumbum_function, initial_guess, method='BFGS')

print("Оптимизированные параметры:", result.x)
print("Минимальные знаечния функции:", result.fun)
print("Кол-во итераций:", result.nit)
