# Deep Learning Flower Species Classifier

A complete end-to-end multi-class classification project built using a TensorFlow Neural Network to identify Iris flower species with high accuracy (~98%).

Project Goal: To build, train, and validate a simple Feed-Forward Neural Network that can accurately classify unseen Iris flowers into one of three species: Setosa, Versicolor, or Virginica.

# Results

Metric

Value

Meaning

Test Accuracy

~96% to 100%

The model can correctly identify a new flower species with high confidence.

Model Type

Sequential Neural Network

A simple but effective deep learning structure.

Data Scaling

MinMaxScaler

Ensures fast and efficient model training.

Training Visualization

The model demonstrated successful learning, with accuracy rapidly increasing and loss rapidly decreasing over the 100 epochs.


# Project Contents

File

Description

flower_species_classification.py

The main script containing all data preprocessing, model building, training, saving, and plotting.

requirements.txt

Lists all necessary Python libraries.

flower_classifier_model.keras

The trained model weights and architecture (saved after running the script).

model_training_curves.png

A visual confirmation that the model learned successfully.

# How to Run the Project

1. Setup Environment

Clone the repository and install the dependencies:

# Install the necessary libraries
pip install -r requirements.txt


2. Execute the Script

Run the main file from your terminal:

python flower_species_classification.py


# Output

The script will:

Load and scale the data.

Train the model for 100 epochs.

Print the final Test Accuracy.

Save the trained model to flower_classifier_model.keras.

Generate and save the training curves plot to model_training_curves.png.

Technologies Used: Python, TensorFlow/Keras, Pandas, NumPy, Matplotlib.
