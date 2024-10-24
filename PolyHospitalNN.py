import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
import numpy as np

# Define the Enhanced Neural Network model with choice of activation function
class EnhancedNeuralNetwork(nn.Module):
    def __init__(self, input_size, activation='elu'):
        super(EnhancedNeuralNetwork, self).__init__()

        # Choose the activation function based on the input
        if activation == 'swish':
            self.activation = nn.SiLU()
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

def train_neural_network_on_synthetic_data_with_polynomial(data_file, chunk_size=1000, epochs=500, learning_rate=0.005, poly_degree=2, activation='elu', weight_decay=1e-14):
    # Load synthetic dataset
    data = pd.read_csv(data_file)

    categorical_columns = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'glucose_test', 'A1Ctest', 'change', 'diabetes_med']
    target_column = 'readmitted'
    numerical_columns = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']

    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    label_encoder = LabelEncoder()
    one_hot_encoder.fit(data[categorical_columns])
    label_encoder.fit(data[target_column])

    scaler = StandardScaler()
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)

    X_numerical = data[numerical_columns].values
    X_numerical_poly = poly.fit_transform(scaler.fit_transform(X_numerical))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_numerical_poly.shape[1] + one_hot_encoder.transform(data[categorical_columns][:1]).shape[1]

    model = EnhancedNeuralNetwork(input_size=input_size, activation=activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        data = data.sample(frac=1).reset_index(drop=True)
        epoch_accuracies = []
        epoch_losses = []

        for start_row in range(0, len(data), chunk_size):
            end_row = min(start_row + chunk_size, len(data))
            chunk = data[start_row:end_row]

            encoded_categorical = one_hot_encoder.transform(chunk[categorical_columns])
            X_numerical = chunk[numerical_columns].values
            X_numerical_poly = poly.transform(scaler.transform(X_numerical))
            X_chunk = np.hstack((X_numerical_poly, encoded_categorical))
            y_chunk = label_encoder.transform(chunk[target_column])

            X_tensor = torch.tensor(X_chunk, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_chunk, dtype=torch.long).to(device)

            model.train()
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = loss_function(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                predictions = model(X_tensor)
                _, predicted_labels = torch.max(predictions, 1)
                accuracy = accuracy_score(y_tensor.cpu(), predicted_labels.cpu())
                epoch_accuracies.append(accuracy)
                epoch_losses.append(loss.item())

        overall_epoch_accuracy = np.mean(epoch_accuracies)
        average_loss = np.mean(epoch_losses)

        print(f"Overall Accuracy after Epoch {epoch + 1}: {overall_epoch_accuracy * 100:.2f}% - Average Loss: {average_loss:.4f}")

    return model, poly, one_hot_encoder, scaler

def test_model_on_original_data_with_polynomial(model, original_data_file, poly, one_hot_encoder, scaler):
    original_data = pd.read_csv(original_data_file)
    categorical_columns = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'glucose_test', 'A1Ctest', 'change', 'diabetes_med']
    numerical_columns = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']
    
    X_numerical = original_data[numerical_columns].values
    X_numerical_poly = poly.transform(scaler.transform(X_numerical))
    encoded_categorical = one_hot_encoder.transform(original_data[categorical_columns])
    X_original = np.hstack((X_numerical_poly, encoded_categorical))

    label_encoder = LabelEncoder()
    y_original = label_encoder.fit_transform(original_data['readmitted'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_original, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_original, dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted_labels = torch.max(outputs, 1)

    original_accuracy = accuracy_score(y_tensor.cpu(), predicted_labels.cpu())
    print(f"Accuracy on Original Dataset: {original_accuracy * 100:.2f}%")

# Example usage of the training and testing functions
synthetic_data_file = 'synthetic_augmented_dataset.csv'
original_data_file = 'hospital_readmissions.csv'

# Train the model using the synthetic dataset with polynomial features and regularization
trained_model, poly, one_hot_encoder, scaler = train_neural_network_on_synthetic_data_with_polynomial(
    synthetic_data_file,
    chunk_size=1000,
    epochs=500,
    learning_rate=0.005,
    poly_degree=2,
    activation='elu',
    weight_decay=1e-14
)

# Test the trained model on the original dataset
test_model_on_original_data_with_polynomial(
    model=trained_model,
    original_data_file=original_data_file,
    poly=poly,
    one_hot_encoder=one_hot_encoder,
    scaler=scaler
)