import cv2
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.preprocessing.image as ImageDataGenerator
import shutil
from sklearn.model_selection import train_test_split

def standardize_images(input_dir, output_dir, target_size=(255, 255)):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of all image files in input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for img_file in image_files:
        try:
            # Construct full file paths
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, img_file)
            
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

#inp='C:/Users/Francisco/Desktop/Projetos/Coffee_Disease/Ferrugem'
#standardize_images(inp, 'C:/Users/Francisco/Desktop/Projetos/Coffee_Disease/teste')

def Separate_Data(dataset_dir, train_dir, test_dir, test_size = 0.1):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for classe in os.listdir(dataset_dir):
     class_dir = os.path.join(dataset_dir, classe)
     if not os.path.isdir(class_dir):
         continue  # pular se não for diretório
     imagens = os.listdir(class_dir)

     # Separar treino e teste
     train_imgs, test_imgs = train_test_split(imagens, test_size=test_size, random_state=42)
     # Criar diretórios de saída
     os.makedirs(os.path.join(train_dir, classe), exist_ok=True)
     os.makedirs(os.path.join(test_dir, classe), exist_ok=True)

     # Copiar arquivos de treino
     for img in train_imgs:
         src = os.path.join(class_dir, img)
         dst = os.path.join(train_dir, classe, img)
         shutil.copy2(src, dst)

     # Copiar arquivos de teste
     for img in test_imgs:
         src = os.path.join(class_dir, img)
         dst = os.path.join(test_dir, classe, img)
         shutil.copy2(src, dst)

     print(f"Classe '{classe}': {len(train_imgs)} treino, {len(test_imgs)} teste.")

    print("Separação concluída com sucesso!")

Separate_Data('Dataset/','treino/','teste/')