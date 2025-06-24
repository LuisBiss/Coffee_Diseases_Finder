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
def gerador_teste(inputshape=(255, 255, 3), target_size=(255, 255), test_dir='test/'):
    test_datagen = ImageDataGenerator(rescale=1./inputshape[1])
    gerador_teste = test_datagen.flow_from_directory(
    test_dir,
    batch_size=16,
    target_size=target_size,
    class_mode='sparse',
    shuffle=False  # Important for evaluation
    )
    return gerador_teste

def gerador_treino(inputshape=(255, 255, 3), target_size=(255, 255), train_dir='train/'):
    train_datagen = ImageDataGenerator(
        rescale=1./inputshape[1],  # Rescale pixel values
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )
    gerador_treino = train_datagen.flow_from_directory(
        train_dir,
        batch_size=16,
        target_size=target_size,
        class_mode='sparse',
    )
    return gerador_treino

def gerador_validacao(inputshape=(255, 255, 3), target_size=(255, 255), val_dir='val/'):
    val_datagen = ImageDataGenerator(rescale=1./inputshape[1])  # Rescale pixel values
    # Gerador de validação
    gerador_validacao = val_datagen.flow_from_directory(
        val_dir,
        batch_size=16,
        target_size=target_size,
        class_mode='sparse',
    )
    return gerador_validacao


# Modelos de RNA
def Simple_RNA_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
def RNA_DenseNet(input_shape, num_classes):  
    #Transferlearning
    base_model = tf.keras.applications.DenseNet121(
    weights='imagenet',  # Use pre-trained weights
    include_top=False,   # Não inserir a camada de classificação final
    input_shape=input_shape,
    )
    base_model.trainable = False  # Não treinar na fazer de transfer learning
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(), #Maybe trocar por Flatten BIG MAYBE
        tf.keras.layers.BatchNormalization(), #overfitting
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5), #overfitting
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
def Multi_layers_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
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
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
def Anti_Overfitting_model(input_shape, num_classes):
    model = tf.keras.Sequential([ 
        # Input Augmentation (applied during inference too)
        # useless as already done in gerador_treino
        #layers.RandomRotation(0.1),
        #layers.RandomZoom(0.1),
        #layers.RandomContrast(0.1),

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
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
''''''
# Utilitários p/ o modelo
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',       # what to monitor
        patience=50,               # how many epochs to wait before stopping
        restore_best_weights=True # restore the best model (not the last one)
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=50
    )
]
def get_optimizer():
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,  # Start at 0.001
        decay_steps=10000,           # Apply decay every 10k steps (≈ epochs if batch_size=32)
        decay_rate=0.9,              # Multiply LR by 0.9 after each decay step
        staircase=True               # Discretize decay (neat jumps vs. continuous)
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return optimizer

# Get predictions
def get_predictions(model, gerador_teste, history):
    y_pred = model.predict(gerador_teste)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = gerador_teste.classes
    # Extract accuracy values from history
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(train_accuracy) + 1)
    return y_true, y_pred_classes, train_accuracy, val_accuracy, epochs

def predict_one(model, gerador_teste, class_names=None):
    # Obter uma imagem e seu rótulo verdadeiro do gerador
    x_batch, y_batch = next(iter(gerador_teste))
    image = x_batch[0]
    true_label = np.argmax(y_batch[0]) if y_batch.ndim > 1 else y_batch[0]

    # Realizar a predição
    y_pred = model.predict(np.expand_dims(image, axis=0)) #Expande as dimensões Aqui
    pred_label = np.argmax(y_pred, axis=1)[0]
    confidence = np.max(y_pred) * 100

    # Nome das classes se disponível
    if class_names is None and hasattr(gerador_teste, 'class_indices'):
        class_names = list(gerador_teste.class_indices.keys())

    # Plotar a imagem com os rótulos
    plt.imshow(image)
    plt.axis('off')
    true_label = int(true_label)
    pred_label = int(pred_label)
    title = f"Real: {class_names[true_label] if class_names else true_label} | Predição: {class_names[pred_label] if class_names else pred_label}" f"({confidence:.2f}%)"
    plt.title(title)
    plt.show()