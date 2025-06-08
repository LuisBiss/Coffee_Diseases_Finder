import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt

data_generator =  ImageDataGenerator(rescale=1/255.0)
gerador_treino = data_generator.flow_from_directory('treino/', batch_size=16, target_size=(255,255), class_mode = 'sparse')
gerador_teste = data_generator.flow_from_directory('teste/', batch_size=16, target_size=(255,255), class_mode = 'sparse')

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(255, 255, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(gerador_treino.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(gerador_treino, epochs=10)

# para o teste geral
print(gerador_teste.class_indices)
plt.imshow(gerador_teste[0][0][0])
print(model.predict(gerador_teste[0][0][0]))