import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle

# Load the CSV
csv_path = 'water_treatment_data.csv'  # Change this to the appropriate file path
data = pd.read_csv(csv_path, sep=';')

# Separate the output (first column) and the input features (remaining columns)
output = data.iloc[:, 0]  # First column as the output
input_features = data.iloc[:, 1:]  # Remaining columns as input features

# Convert to numeric values (if necessary)
output = output.astype(float)
input_features = input_features.astype(float)

# K-Fold Cross Validation setup
k_folds = 5  # Define the number of folds
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Store metrics for each fold
fold_losses = []
fold_maes = []
fold_r2s = []

# Cross-validation loop
for fold, (train_index, val_index) in enumerate(kf.split(input_features), start=1):
    print(f"\nTraining fold {fold}/{k_folds}")

    # Split the data into train and validation sets
    X_train, X_val = input_features.iloc[train_index], input_features.iloc[val_index]
    y_train, y_val = output.iloc[train_index], output.iloc[val_index]

    # Standardize the input features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Create the model
    model = Sequential([
        Dense(6, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(1, activation=None)  # Linear activation for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=32,
        verbose=0,  # Set verbose to 0 to avoid printing epoch details
        callbacks=[early_stopping]
    )

    # Evaluate the model
    loss, mae = model.evaluate(X_val, y_val, verbose=0)
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)

    # Store metrics for this fold
    fold_losses.append(loss)
    fold_maes.append(mae)
    fold_r2s.append(r2)

    print(f"Fold {fold} results - Loss: {loss:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

# Calculate and print the average performance across all folds
print("\nAverage performance across all folds:")
print(f"Average Loss: {np.mean(fold_losses):.4f}")
print(f"Average MAE: {np.mean(fold_maes):.4f}")
print(f"Average R^2: {np.mean(fold_r2s):.4f}")


# Display calculated weights and biases
def generate_db_format(weights_layer1, biases_layer1, weights_layer2, biases_layer2, column_names):
    db_output = []

    # Weights and biases of the hidden layer
    for i, neuron_weights in enumerate(weights_layer1.T):  # Transpose to group by neurons
        for j, weight in enumerate(neuron_weights):
            db_output.append(f"W_{j + 1}_{i + 1};\tReal;\t{weight:.4f}")
        db_output.append(f"B_{i + 1};\tReal;\t{biases_layer1[i]:.4f}")

    # Weights and biases of the output layer
    for i, output_weight in enumerate(weights_layer2.flatten()):  # Weights towards the single output neuron
        db_output.append(f"W_{i + 1}_O;\tReal;\t{output_weight:.4f}")
    db_output.append(f"B_O;\tReal;\t{biases_layer2[0]:.4f}")

    # Mean (MU) and Standard Deviation (Sigma) for each input feature
    for i, col_name in enumerate(column_names):
        db_output.append(f"MU_{col_name};\tReal;\t{scaler.mean_[i]:.4f}")
        db_output.append(f"SIGMA_{col_name};\tReal;\t{scaler.scale_[i]:.4f}")

    # Print the result in Data Block format
    print("\nData Block Format:")
    for line in db_output:
        print(line)

# Extract weights and biases
weights_layer1, biases_layer1 = model.layers[0].get_weights()
weights_layer2, biases_layer2 = model.layers[2].get_weights()

generate_db_format(weights_layer1, biases_layer1, weights_layer2, biases_layer2, data.columns[1:])



# Optionally, save the model and scaler from the last fold
model.save('trained_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Optionally, plot the training history (for the last fold)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training History (Last Fold)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
