import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Tratamento dos dados
test_datagen = ImageDataGenerator(rescale=1./256)
train_datagen = ImageDataGenerator(
    rescale=1./256,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)
gerador_treino = train_datagen.flow_from_directory(
    'treino/',
    batch_size=16,
    target_size=(256, 256),
    class_mode='sparse',
    subset='training',  # This takes the training portion
)
# Validation data (20% of training directory)
gerador_validacao = train_datagen.flow_from_directory(
    'treino/',
    batch_size=16,
    target_size=(256, 256),
    class_mode='sparse',
    subset='validation',  # This takes the validation portion
)
gerador_teste = test_datagen.flow_from_directory(
    'teste/',
    batch_size=16,
    target_size=(256, 256),
    class_mode='sparse',
    shuffle=False  # Important for evaluation
)

# Modelos de RNA
Simple_RNA_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Flatten(input_shape=(256, 256, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(gerador_treino.num_classes, activation='softmax')
])
Multi_layers_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(gerador_treino.num_classes, activation='softmax')
])
Anti_Overfitting_model = tf.keras.Sequential([
        # Input Augmentation (applied during inference too)
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
        kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
        kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
        kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.5),
        layers.Dense(gerador_treino.num_classes, activation='softmax')
    ])
# Utilitários p o modelo
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',       # what to monitor
        patience=3,               # how many epochs to wait before stopping
        restore_best_weights=True # restore the best model (not the last one)
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5
    )
]
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,  # Start at 0.001
    decay_steps=10000,           # Apply decay every 10k steps (≈ epochs if batch_size=32)
    decay_rate=0.9,              # Multiply LR by 0.9 after each decay step
    staircase=True               # Discretize decay (neat jumps vs. continuous)
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compilando e treinando o modelo
Anti_Overfitting_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Anti_Overfitting_model.build(input_shape=(None, 256, 256, 3))  # Build the model with input shape
print("Model Summary:")
Anti_Overfitting_model.summary()
Anti_Overfitting_model.fit(
    gerador_treino,
    epochs=50,
    steps_per_epoch=len(gerador_treino),  # Number of batches per epoch
    validation_data=gerador_validacao,
    validation_steps=len(gerador_validacao),  # Batches for validation
    callbacks=callbacks,
    verbose=1
)
# Display result
# Get predictions
y_pred = Anti_Overfitting_model.predict(gerador_teste)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = gerador_teste.classes

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(gerador_teste.class_indices.keys()))
disp.plot(cmap=plt.cm.Blues)
plt.show()
"""
plt.imshow(image)
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()

print("Predicted class:", predicted_class)
"""