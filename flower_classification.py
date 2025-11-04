# --- IMPORTS ---
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import os
import matplotlib.pyplot as plt  # NEW: For plotting

# -------------------------------------------------------------
# CODE STEP 1: Load and Prepare Data
# -------------------------------------------------------------
# 1. Load the Iris Dataset
iris_data = load_iris(as_frame=True)
df = iris_data.frame
df.rename(columns={'target': 'species_code'}, inplace=True)
df['species'] = df['species_code'].map(lambda x: iris_data.target_names[x])

# -------------------------------------------------------------
# CODE STEP 2: Feature Engineering and Data Splitting
# -------------------------------------------------------------
# 1. Separate Features (X) and Target (y)
X = df.drop(['species_code', 'species'], axis=1)
y_raw = df['species_code']

# 2. Encode the Target (y)
y_one_hot = to_categorical(y_raw)

# 3. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot,
    test_size=0.2,
    random_state=42
)

# 4. Scale the Features (X)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Final Check ---
print("--- Data Split Sizes ---")
print(f"X_train (features for training): {X_train_scaled.shape}")
print(f"y_train (targets for training): {y_train.shape}")
print(f"y_test (first one-hot example): {y_test[0]}")
print(f"X_test (first scaled example): {X_test_scaled[0]}")

# -------------------------------------------------------------
# CODE STEP 3: Define, Compile, and Train the Model
# -------------------------------------------------------------
# Get the number of input features (4: the measurements)
input_shape = X_train_scaled.shape[1]
# Get the number of output classes (3: the species)
output_classes = y_train.shape[1]

# 1. Define the Neural Network Architecture
model = Sequential()
model.add(Dense(units=10, activation='relu', input_dim=input_shape))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=output_classes, activation='softmax'))

# 2. Compile the Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Optional: Print a summary of the model's layers and parameters
print("\n--- Model Summary ---")
model.summary()

# 3. Train the Model
print("\n--- Model Training Started ---")
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=5,
    verbose=1,
    validation_data=(X_test_scaled, y_test)
)
print("--- Model Training Finished ---")

# -------------------------------------------------------------
# CODE STEP 4: Evaluation and Prediction
# -------------------------------------------------------------
# 1. Evaluate the model's performance on the unseen test data
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print("\n--- Model Evaluation ---")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 2. Make predictions on the test set
raw_predictions = model.predict(X_test_scaled, verbose=0)

# 3. Convert probabilities back to class labels (Species)
iris_species_names = ['setosa', 'versicolor', 'virginica']
predicted_classes_indices = np.argmax(raw_predictions, axis=1)
predicted_species_names = [iris_species_names[i] for i in predicted_classes_indices]
true_classes_indices = np.argmax(y_test, axis=1)
true_species_names = [iris_species_names[i] for i in true_classes_indices]

# 4. Display Final Comparison
print("\n--- Final Prediction Comparison (First 10) ---")
results = pd.DataFrame({
    'Predicted Species': predicted_species_names[:10],
    'True Species': true_species_names[:10]
})
print(results)

# -------------------------------------------------------------
# CODE STEP 5: Saving and Reloading the Model
# -------------------------------------------------------------

MODEL_FILEPATH = 'flower_classifier_model.keras'

# 1. Save the model to a single HDF5 file
print(f"\n--- Saving Model to: {MODEL_FILEPATH} ---")
try:
    model.save(MODEL_FILEPATH)
    print("Model successfully saved!")

    # 2. Verify the file was created
    if os.path.exists(MODEL_FILEPATH):
        print("Verification: Model file exists on disk.")

except Exception as e:
    print(f"Error saving model: {e}")

# 3. Reload the model (as a test)
try:
    reloaded_model = load_model(MODEL_FILEPATH)
    print("Model successfully reloaded from disk.")

    # 4. Test the reloaded model's performance
    reloaded_loss, reloaded_accuracy = reloaded_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Reloaded Model Accuracy Check: {reloaded_accuracy * 100:.2f}%")

except Exception as e:
    print(f"Error reloading model: {e}")

# -------------------------------------------------------------
# CODE STEP 6: Visualization
# -------------------------------------------------------------

# Plotting the Training Accuracy and Validation Accuracy
plt.figure(figsize=(12, 5))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over 100 Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# Subplot 2: Loss (Error)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over 100 Epochs')
plt.ylabel('Loss (Error)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# Save the plot to a file before showing it
plt.savefig('model_training_curves.png')

# Show the plot
plt.show()