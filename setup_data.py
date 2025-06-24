import cv2
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.preprocessing.image as ImageDataGenerator
import shutil
import seaborn as sns
from sklearn.model_selection import train_test_split

# Processando as imagens do dataset e criando o diretório de saída
def standardize_images(input_dir, output_dir='output_dir/', target_size=(255,255)):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for classe in os.listdir(input_dir):
     class_dir = os.path.join(input_dir, classe)
     if not os.path.isdir(class_dir):
         continue
     
     class_output_dir = os.path.join(output_dir, classe)
     if not os.path.exists(class_output_dir):
        os.makedirs(class_output_dir)
     # lista de imagens na classe
     image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
     for img_file in image_files:
         try:
             # Path's do arquivo
             input_path = os.path.join(class_dir, img_file)
             output_path = os.path.join(class_output_dir, img_file)
             
             # Read image
             img = cv2.imread(input_path)
             if img is None:
                 print(f"Warning: Could not read image {img_file}. Skipping...")
                 continue
             
             # Resize image
             resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
             
             # Save resized image
             cv2.imwrite(output_path, resized_img)
             print(f"Processed: {img_file}")
         except Exception as e:
             print(f"Error processing {img_file}: {str(e)}")
     print("Imagens padronizadas")

def separate_data(input_dir, train_dir, test_dir,val_dir, val_ratio=0.2, test_ratio=0.1):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for classe in os.listdir(input_dir):
     #Array de imagens da classe
     class_dir = os.path.join(input_dir, classe)
     if not os.path.isdir(class_dir):
         continue  # pular se não for diretório
     imagens = os.listdir(class_dir)

     # Dividir o as imagens em treino e teste
     train_val_imgs, test_imgs = train_test_split(imagens, test_size=test_ratio)
     train_imgs, val_imgs = train_test_split(train_val_imgs, test_size=val_ratio / (1 - test_ratio))

     # Criar diretórios de saída
     os.makedirs(os.path.join(train_dir, classe), exist_ok=True)
     os.makedirs(os.path.join(test_dir, classe), exist_ok=True)
     os.makedirs(os.path.join(val_dir, classe), exist_ok=True)

     # Copiar arquivos p/ cada Diretório
     for img in train_imgs:
         src = os.path.join(class_dir, img)
         dst = os.path.join(train_dir, classe, img)
         shutil.copy2(src, dst)

     for img in test_imgs:
         src = os.path.join(class_dir, img)
         dst = os.path.join(test_dir, classe, img)
         shutil.copy2(src, dst)

     for img in val_imgs:
         src = os.path.join(class_dir, img)
         dst = os.path.join(val_dir, classe, img)
         shutil.copy2(src, dst)

     print(f"Classe '{classe}': {len(train_imgs)} treino, {len(test_imgs)} teste, {len(val_imgs)} validação")

    print("Separação concluída com sucesso!")

#standardize_images('Dataset/', 'output_dir/')
#separate_data('output_dir/','treino/','teste/','validação/')