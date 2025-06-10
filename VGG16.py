import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data_generator =  ImageDataGenerator(rescale=1/255.0)
gerador_treino = data_generator.flow_from_directory('treino/', batch_size=16, target_size=(255,255), class_mode = 'sparse')
gerador_teste = data_generator.flow_from_directory('teste/', batch_size=16, target_size=(255,255), class_mode = 'sparse')

Vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(255,255,3))

average_pooling = tf.keras.layers.GlobalAveragePooling2D()(Vgg16.output)
hidden = tf.keras.layers.Dense(2048, activation='relu')(average_pooling)
pred = tf.keras.layers.Dense(5, activation='softmax')(hidden)
Vgg16_final = tf.keras.models.Model(inputs=Vgg16.input, outputs=pred)

Vgg16_final.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Vgg16_final.summary()

Vgg16_final.fit(gerador_treino, epochs=5)

# Display result
# Get predictions
y_pred = Vgg16.predict(gerador_teste)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = gerador_teste.classes

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(gerador_teste.class_indices.keys()))
disp.plot(cmap=plt.cm.Blues)
plt.show()