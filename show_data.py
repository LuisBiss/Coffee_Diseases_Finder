import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Define the dataset path
dataset_path = 'C:/Users/Francisco/Desktop/Projetos/Coffee_Diseases_Finder/Oficiialdata'

# Define the splits (train, validation, test)
splits = ['treino', 'validação', 'teste']
classes = ['Ferrugem', 'Fosforo', 'Healthy', 'Mineiro', 'Phoma', 'Pulga_Vermelha']  # Adjust class names if needed
def data_distribuition(dataset_path, splits, classes):
    """
    Function to visualize the distribution of images across different classes and splits.
    It counts the number of images in each class for train, validation, and test splits,
    and then plots a bar chart to show the distribution.
    """
    # Store counts for each split and class
    counts = {split: [] for split in splits}

    # Count images in each directory
    for split in splits:
        for cls in classes:
            cls_path = os.path.join(dataset_path, split, cls)
            print(f"Checking path: {cls_path}")  # Debugging line to check paths
            if os.path.exists(cls_path):
                num_images = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                counts[split].append(num_images)
                print(f"Count for {split} - {cls}: {num_images}")
            else:
                counts[split].append(0)  # In case a class folder is missing

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.25
    index = range(len(classes))

    # Plot bars for each split
    for i, split in enumerate(splits):
        ax.bar([x + i * bar_width for x in index], counts[split], bar_width, label=split)

    # Customize plot
    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Images')
    ax.set_title('Image Distribution Across Classes and Splits')
    ax.set_xticks([x + bar_width for x in index])
    ax.set_xticklabels(classes)
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_curves(epochs, train_accuracy, val_accuracy):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
    plt.title('Training vs. Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()



# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred_classes, gerador_teste):
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(gerador_teste.class_indices.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()