import sys
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from setup_data import standardize_images, separate_data
from app import (
    Simple_RNA_model,
    RNA_DenseNet,
    Multi_layers_model,
    Anti_Overfitting_model,
    gerador_teste,
    gerador_treino,
    gerador_validacao,
    get_optimizer,
    get_predictions,
    predict_one
)
from show_data import (
    data_distribuition,
    plot_curves,
    plot_confusion_matrix,
)
def resolve_path(path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, os.path.expanduser(path)))

def main():
    print("--- Processamento de dados ---")
    skip_processing = input("Pular processamento de dados? (s/n): ").strip().lower()

    if skip_processing != 's':
        input_dir = resolve_path(input("Insira o path do Dataset: ") or "Dataset/")
        output_dir = resolve_path(input("Insira o path do diretório para tratamento de imagems: ") or "output_dir/")
   
        try:
            target_size_input = input("Insira as dimensões da imagem (default 255 255): ")
            if target_size_input:
                target_size = tuple(map(int, target_size_input.split()))
            else:
                target_size = (255, 255)
        except ValueError:
            print("Tamamanho invalido. Using default (255, 255).")
            target_size = (255, 255)
        standardize_images(input_dir, output_dir, target_size=target_size)
        train_dir = resolve_path(input("Insira a pasta de treinamento: ") or "train/")
        test_dir = resolve_path(input("Insira a pasta de teste: ") or "test/")
        val_dir = resolve_path(input("Insira a pasta de validação: ") or "val/")
    
        try:
            val_ratio = float(input("Entre com o ratio de validação (default 0.2): ") or 0.2)
            test_ratio = float(input("Entre com o ratio de teste(default 0.1): ") or 0.1)
        except ValueError:
            print("Invalid ratio.")
            val_ratio = 0.2
            test_ratio = 0.1
        separate_data(output_dir, train_dir, test_dir, val_dir, val_ratio=val_ratio, test_ratio=test_ratio)
        data_distribuition(output_dir, ['treino', 'validação', 'teste'], ['Ferrugem', 'Fosforo', 'Healthy', 'Mineiro', 'Phoma', 'Pulga_Vermelha'])
    else:
        input_dir = resolve_path(input("Insira o path do Dataset já processado: "))
        output_dir = input_dir
        target_size = (255, 255)
        train_dir = resolve_path(input("Insira a pasta de treinamento: "))
        test_dir = resolve_path(input("Insira a pasta de teste: "))
        val_dir = resolve_path(input("Insira a pasta de validação: "))
    print("\n--- Seleção do Modelo ---")
    print("Escolha o modelo:")
    models = {
        "1": ("Simple_RNA_model", Simple_RNA_model),
        "2": ("RNA_DenseNet", RNA_DenseNet),
        "3": ("Multi_layers_model", Multi_layers_model),
        "4": ("Anti_Overfitting_model", Anti_Overfitting_model)
    }
    for key, (name, _) in models.items():
        print(f"{key}: {name}")

    choice = input("Escolha o modelo: ").strip()
    if choice not in models:
        print("Invalido.")
        sys.exit(1)

    input_shape = (*target_size, 3)
    num_classes = sum(1 for entry in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, entry)))

    model_func = models[choice][1]
    model = model_func(input_shape, num_classes)
    print("\nModelo criado:")
    model.summary()
    optimizer = get_optimizer()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    gerador_treino_local = gerador_treino(inputshape=input_shape, target_size=target_size, train_dir=train_dir)
    gerador_validacao_local= gerador_validacao(inputshape=input_shape, target_size=target_size, val_dir=val_dir)
    gerador_teste_local= gerador_teste(inputshape=input_shape, target_size=target_size, test_dir=test_dir)
    # Treinamento
    epochs = input("Insira o número de épocas (default 15): ")
    try:
        epochs = int(epochs) if epochs else 15
    except ValueError:
        print("Entrada inválida. Usando valor padrão de 15 épocas.")
        epochs = 15
    history = model.fit(
        gerador_treino_local,
        epochs=epochs,
        steps_per_epoch=len(gerador_treino_local),  # Number of batches per epoch
        validation_data=gerador_validacao_local,
        validation_steps=len(gerador_validacao_local),  # Batches for validation
        verbose=1
    )
    y_true, y_pred_classes, train_accuracy, val_accuracy, epochs = get_predictions(model, gerador_teste_local, history)
    plot_curves(epochs, train_accuracy, val_accuracy)
    plot_confusion_matrix(y_true, y_pred_classes, gerador_teste_local)
    predict_one(model, gerador_teste_local)

if __name__ == "__main__":
    main()