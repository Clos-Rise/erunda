import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

X = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(SimpleRNN(units=2, input_shape=(1, 2), activation='tanh'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=1000, verbose=0)

predictions = model.predict(X)
print("Предсказание:")
print(predictions)
