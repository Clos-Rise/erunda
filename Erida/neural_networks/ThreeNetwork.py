import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model

input_shape_1 = 10
input_shape_2 = 15
input_shape_3 = 20

input_1 = Input(shape=(input_shape_1,))
x1 = Dense(64, activation='relu')(input_1)
x1 = Dense(32, activation='relu')(x1)
output_1 = Dense(1, activation='linear')(x1)
neural_1 = Model(inputs=input_1, outputs=output_1)

input_2 = Input(shape=(input_shape_2,))
x2 = Dense(64, activation='relu')(input_2)
x2 = Dense(32, activation='relu')(x2)
output_2 = Dense(1, activation='linear')(x2)
neural_2 = Model(inputs=input_2, outputs=output_2)

input_3 = Input(shape=(input_shape_3,))
x3 = Dense(64, activation='relu')(input_3)
x3 = Dense(32, activation='relu')(x3)
output_3 = Dense(1, activation='linear')(x3)
neural_3 = Model(inputs=input_3, outputs=output_3)

combined = Concatenate()([neural_1.output, neural_2.output, neural_3.output])
z = Dense(10, activation='relu')(combined)
main_output = Dense(1, activation='linear')(z)

model = Model(inputs=[neural_1.input, neural_2.input, neural_3.input], outputs=main_output)

model.compile(optimizer='adam', loss='mse')
model.summary()
