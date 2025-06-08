import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt
#print(teste)

data_generator =  ImageDataGenerator(rescale=1/255.0)
gerador_treino = data_generator.flow_from_directory('treino/', batch_size=16, target_size=(255,255), class_mode = 'sparse')
gerador_teste = data_generator.flow_from_directory('teste/', batch_size=16, target_size=(255,255), class_mode = 'sparse')

model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(255,255,3))

average_pooling = tf.keras.layers.GlobalAveragePooling2D()(model.output)
hidden = tf.keras.layers.Dense(2048, activation='relu')(average_pooling)
pred = tf.keras.layers.Dense(5, activation='softmax')(hidden)
model_final = tf.keras.models.Model(inputs=model.input, outputs=pred)

model_final.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_final.summary()

model_final.fit(gerador_treino, epochs=50)

# para o teste geral
print(gerador_teste.class_indices)
plt.imshow(gerador_teste[0][0][0])
print(model_final.predict(gerador_teste[0][0][0].reshape(255,255,3)))