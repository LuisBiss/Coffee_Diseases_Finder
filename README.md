Coffee Leaf Disease Classification using Neural Networks

This project implements a neural network system for classifying diseases in coffee leaves. The system processes images of coffee leaves, trains various neural network models, and provides predictions for disease classification. It can standardize images use several types of Neural Network chosse by you

Features:

    Image preprocessing and standardization

    Data separation into training, validation, and test sets

    Multiple neural network architectures to choose from

    Training and evaluation metrics

    Confusion matrix visualization

    Single image prediction capability

## 🚀 Getting Started

Follow the steps below to set up and run the project:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/coffee-leaf-disease-classification.git
cd coffee-leaf-disease-classification
```
### 2. Create and activate a virtual environment

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
On Linux/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the application
```bash
python main.py
```

## 📁 Dataset Structure
The dataset should be told when asked by the terminal at the root of the repository, and structured as follows:

Dataset/

├── Ferrugem/

├── Fosforo/

├── Healthy/

├── Mineiro/

├── Phoma/

└── Pulga_Vermelha/

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
