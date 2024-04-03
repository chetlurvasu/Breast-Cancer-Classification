# Breast-Cancer-Classification

This project aims to classify breast cancer tumors as either malignant or benign using machine learning techniques, particularly a neural network implemented with TensorFlow and Keras.

Dataset

The dataset used in this project is obtained from the sklearn library and consists of various features extracted from breast cancer biopsies, along with corresponding labels indicating tumor types (malignant or benign).

Data Preprocessing

The dataset is loaded into a pandas DataFrame for exploration and preprocessing. Missing values are checked and handled appropriately. Standardization is performed using sklearn's StandardScaler to ensure uniform scaling across features.

Model Architecture

The neural network model comprises a input layer, a hidden layer with ReLU activation function, and an output layer with sigmoid activation function for binary classification. The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function.

Training

The model is trained on the preprocessed data, with a validation split for monitoring performance during training. Training is performed for a specified number of epochs.

Evaluation

Model performance is evaluated on the test set using accuracy as the evaluation metric.

Prediction

The trained model can be used to predict the tumor type (malignant or benign) for new input data. Input data is standardized before making predictions.

Dependencies
1. Python 3.2
2. TensorFlow
3. Keras
4. numpy
5. pandas
6. scikit-learn


Usage
1. Clone the repository.
2. Install the dependencies listed in requirements.txt.
3. Train the model using train.py.
4. Evaluate the model using evaluate.py.
5. Make predictions using predict.py.
   
Feel free to contribute or provide feedback!
