import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import json
import numpy as np

with open('dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = [item['text'] for item in data]
authors = [item['author'] for item in data]

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

data_1 = pad_sequences(sequences, maxlen=100)

author_tokenizer = Tokenizer()
author_tokenizer.fit_on_texts(authors)
author_sequences = author_tokenizer.texts_to_sequences(authors)
author_index = author_tokenizer.word_index

author_labels = np.array([seq[0] for seq in author_sequences])
num_classes = len(author_index) + 1
author_labels = to_categorical(author_labels, num_classes=num_classes)
input_shape_1 = data_1.shape[1]

input_1 = Input(shape=(input_shape_1,))
x1 = Embedding(input_dim=len(word_index) + 1, output_dim=128)(input_1)
x1 = Flatten()(x1)
x1 = Dense(128, activation='relu')(x1)
x1 = Dense(64, activation='relu')(x1)
x1 = Dense(32, activation='relu')(x1)
output_1 = Dense(num_classes, activation='softmax')(x1)
neural_1 = Model(inputs=input_1, outputs=output_1)
model = Model(inputs=neural_1.input, outputs=output_1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(data_1, author_labels, epochs=10, batch_size=32)

model.save('model.keras')
