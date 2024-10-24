import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# Define a custom Swish activation function class
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Define the Enhanced Neural Network model with choice of activation function
class EnhancedNeuralNetwork(nn.Module):
    def __init__(self, input_size, activation='elu'):
        super(EnhancedNeuralNetwork, self).__init__()

        # Choose the activation function based on the input
        if activation == 'swish':
            self.activation = Swish()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        else:
            self.activation = nn.ReLU()

        # Define the layers with more complexity and dropout for regularization
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            self.activation,
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            self.activation,
            nn.Dropout(0.15)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Dropout(0.15)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            self.activation,
            nn.Dropout(0.15)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            self.activation,
            nn.Dropout(0.15)
        )
        self.fc6 = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            self.activation,
            nn.Dropout(0.15)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            self.activation,
            nn.Dropout(0.15)
        )
        self.fc8 = nn.Sequential(
            nn.Linear(16, 16),
            self.activation
        )
        self.output = nn.Linear(16, 2)

        # Initialize weights using He Initialization
        self.initialize_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.output(x)
        return x

    def initialize_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

def preprocess_original_data(data, categorical_columns, numerical_columns, encoder, scaler):
    # Encode categorical columns using the previously fitted OneHotEncoder
    encoded_categorical = encoder.transform(data[categorical_columns])

    # Standardize numerical columns using the previously fitted StandardScaler
    X_numerical = data[numerical_columns].values
    X_numerical_scaled = scaler.transform(X_numerical)

    # Combine numerical and categorical data
    X_combined = np.hstack((X_numerical_scaled, encoded_categorical))
    return X_combined

def train_neural_network_on_synthetic_data(synthetic_data_file, epochs=5, chunk_size=1000, learning_rate=0.005, activation='elu'):
    # Load the synthetic dataset (already preprocessed)
    data = pd.read_csv(synthetic_data_file)
    
    # Identify columns for input and target
    target_column = 'readmitted'
    input_columns = data.columns.drop(target_column)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the neural network model with chosen activation function
    input_size = len(input_columns)
    model = EnhancedNeuralNetwork(input_size=input_size, activation=activation).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-14)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}")
        
        # Shuffle the dataset at the start of each epoch
        data = data.sample(frac=1).reset_index(drop=True)

        epoch_accuracies = []
        epoch_losses = []

        for start_row in range(0, len(data), chunk_size):
            # Select a chunk of data
            end_row = min(start_row + chunk_size, len(data))
            chunk = data.iloc[start_row:end_row]

            # Separate features and target
            X_chunk = chunk[input_columns].values
            y_chunk = chunk[target_column].values

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
                epoch_losses.append(loss.item())

        # Calculate and print overall accuracy and average loss for the epoch
        overall_epoch_accuracy = np.mean(epoch_accuracies)
        average_loss = np.mean(epoch_losses)

        print(f"Overall Accuracy after Epoch {epoch + 1}: {overall_epoch_accuracy * 100:.2f}% - Average Loss: {average_loss:.4f}")

    return model

def test_model_on_original_data(model, original_data_file, categorical_columns, numerical_columns):
    # Load the original dataset
    original_data = pd.read_csv(original_data_file)

    # Initialize and fit OneHotEncoder and StandardScaler on the original dataset
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    scaler = StandardScaler()

    one_hot_encoder.fit(original_data[categorical_columns])
    scaler.fit(original_data[numerical_columns])

    # Preprocess the original dataset
    X_original = preprocess_original_data(original_data, categorical_columns, numerical_columns, one_hot_encoder, scaler)

    # Encode target column using LabelEncoder
    label_encoder = LabelEncoder()
    y_original = label_encoder.fit_transform(original_data['readmitted'])

    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_original, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_original, dtype=torch.long).to(device)

    # Evaluate the model on the original dataset
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted_labels = torch.max(outputs, 1)

    # Calculate accuracy
    original_accuracy = accuracy_score(y_tensor.cpu(), predicted_labels.cpu())
    print(f"Accuracy on Original Dataset: {original_accuracy * 100:.2f}%")

# Example usage of the training and testing functions
synthetic_data_file = 'synthetic_augmented_dataset.csv'
original_data_file = 'hospital_readmissions.csv'

# Train the model using the synthetic dataset
trained_model = train_neural_network_on_synthetic_data(
    synthetic_data_file,
    epochs=5000,
    chunk_size=1000,
    learning_rate=0.005,
    activation='elu'
)

# Define columns for preprocessing the original dataset
categorical_columns = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'glucose_test', 'A1Ctest', 'change', 'diabetes_med']
numerical_columns = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']

# Test the model on the original dataset
test_model_on_original_data(
    model=trained_model,
    original_data_file=original_data_file,
    categorical_columns=categorical_columns,
    numerical_columns=numerical_columns
)







