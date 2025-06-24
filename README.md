Coffee Leaf Disease Classification using Neural Networks

https://via.placeholder.com/800x400.png?text=Coffee+Leaf+Disease+Examples

This project implements a neural network system for classifying diseases in coffee leaves. The system processes images of coffee leaves, trains various neural network models, and provides predictions for disease classification.
Features

    Image preprocessing and standardization

    Data separation into training, validation, and test sets

    Multiple neural network architectures to choose from

    Training and evaluation metrics

    Confusion matrix visualization

    Single image prediction capability

Installation
1. Clone the repository
bash

git clone https://github.com/yourusername/coffee-leaf-disease-classification.git
cd coffee-leaf-disease-classification

2. Create a virtual environment
bash

python -m venv venv

3. Activate the virtual environment

    Windows:
    bash

venv\Scripts\activate

macOS/Linux:
bash

    source venv/bin/activate

4. Install dependencies
bash

pip install -r requirements.txt

Usage
1. Prepare your dataset

Organize your coffee leaf images in the following directory structure:
text

Dataset/
├── Ferrugem/
├── Fosforo/
├── Healthy/
├── Mineiro/
├── Phoma/
└── Pulga_Vermelha/

2. Run the main script
bash

python main.py

3. Follow the interactive prompts

The program will guide you through:

    Data Processing:

        Input directory path (default: Dataset/)

        Output directory for processed images (default: output_dir/)

        Image dimensions (default: 255x255)

        Validation and test ratios (default: 0.2 and 0.1)

    Model Selection:

        Choose from four neural network architectures:

            Simple RNA Model

            RNA DenseNet

            Multi-layer Model

            Anti-Overfitting Model

        Select optimizer (default: Adam)

    Training:

        Specify number of epochs (default: 15)

    Evaluation:

        View accuracy curves

        Examine confusion matrix

        Test single image prediction

4. Skip data processing (optional)

If you already have processed data, you can skip the data processing step by entering s when prompted.
Project Structure
text

├── main.py              # Main entry point
├── app.py               # Neural network model implementations
├── setup_data.py        # Data preprocessing and separation
├── show_data.py         # Visualization functions
├── requirements.txt     # Dependencies
├── README.md            # This file
└── Dataset/             # Sample directory structure (not included)

Requirements

    Python 3.7+

    TensorFlow 2.x

    NumPy

    Matplotlib

    scikit-learn

    Pillow

Sample Outputs

After training completes, the program will display:

    Training and validation accuracy curves

    Confusion matrix for model performance

    Single image prediction example

Contributing

Contributions are welcome! Please open an issue or submit a pull request.