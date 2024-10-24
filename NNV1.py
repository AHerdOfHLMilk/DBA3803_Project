import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
import numpy as np

# Define the PyTorch neural network model
class NeuralNetworkWithPoly(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetworkWithPoly, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)  # First hidden layer with 50 neurons
        self.fc2 = nn.Linear(50, 20)          # Second hidden layer with 20 neurons
        self.output = nn.Linear(20, 2)        # Output layer (2 classes: readmitted or not)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.output(x)
        return x

def train_with_polynomial_features(data, chunk_size=1000, epochs=5, learning_rate=0.001, poly_degree=2):
    # Convert categorical columns to numeric using OneHotEncoder or LabelEncoder
    categorical_columns = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'glucose_test', 'A1Ctest', 'change', 'diabetes_med']
    target_column = 'readmitted'

    # Define OneHotEncoder and LabelEncoder
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    label_encoder = LabelEncoder()

    # Fit encoders on the full dataset for consistency across chunks
    one_hot_encoder.fit(data[categorical_columns])
    label_encoder.fit(data[target_column])

    # Initialize StandardScaler and PolynomialFeatures for numerical features
    scaler = StandardScaler()
    numerical_columns = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)

    # Fit PolynomialFeatures on the dataset's numerical columns using NumPy array to avoid warnings
    poly.fit(data[numerical_columns].values)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_entries_processed = 0

    # Define the neural network model
    input_size = poly.transform(data[numerical_columns].values[:1]).shape[1] + one_hot_encoder.transform(data[categorical_columns][:1]).shape[1]
    model = NeuralNetworkWithPoly(input_size=input_size).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}")
        epoch_accuracies = []

        for start_row in range(0, len(data), chunk_size):
            # Select a chunk of data
            end_row = min(start_row + chunk_size, len(data))
            chunk = data[start_row:end_row]

            # Preprocess the chunk
            encoded_categorical = one_hot_encoder.transform(chunk[categorical_columns])
            X_numerical = chunk[numerical_columns].values

            # Standardize numerical features and generate polynomial features
            X_numerical_scaled = scaler.fit_transform(X_numerical)
            X_numerical_poly = poly.transform(X_numerical_scaled)

            # Combine polynomial numerical features with encoded categorical features
            X_chunk = np.hstack((X_numerical_poly, encoded_categorical))

            # Encode target column
            y_chunk = label_encoder.transform(chunk[target_column])

            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X_chunk, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_chunk, dtype=torch.long).to(device)

            # Training step
            model.train()
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(X_tensor)
            loss = loss_function(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            # Evaluate the model on the current chunk
            model.eval()
            with torch.no_grad():
                predictions = model(X_tensor)
                _, predicted_labels = torch.max(predictions, 1)
                accuracy = accuracy_score(y_tensor.cpu(), predicted_labels.cpu())
                epoch_accuracies.append(accuracy)

            # Update the total number of entries processed and print current progress
            total_entries_processed += (end_row - start_row)
            print(f"Epoch {epoch + 1} - Processed {total_entries_processed} entries - Accuracy: {accuracy * 100:.2f}%")

        # Calculate and print overall accuracy for the epoch
        overall_epoch_accuracy = np.mean(epoch_accuracies)
        print(f"Overall Accuracy after Epoch {epoch + 1}: {overall_epoch_accuracy * 100:.2f}%\n")

    return model, overall_epoch_accuracy

data = pd.read_csv('hospital_readmissions.csv')

# Example call to function (Assume 'data' is the DataFrame containing your dataset)
trained_model, overall_accuracy = train_with_polynomial_features(data, chunk_size=1000, epochs=5, learning_rate=0.005, poly_degree=5)