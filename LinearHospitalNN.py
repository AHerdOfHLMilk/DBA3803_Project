import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os  # Import the os module
from scipy import stats  # Importing stats for statistical tests

# Define columns for preprocessing the original dataset
categorical_columns = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'glucose_test', 'A1Ctest', 'change', 'diabetes_med']
numerical_columns = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']

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

def compare_datasets(original_data, synthetic_data):
    """
    Compare original and synthetic datasets visually and statistically.
    
    Args:
    - original_data (DataFrame): The original dataset.
    - synthetic_data (DataFrame): The synthetic dataset.
    """
    # Preprocess the datasets
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    scaler = StandardScaler()

    # Fit encoders on the categorical and numerical columns
    one_hot_encoder.fit(original_data[categorical_columns])
    scaler.fit(original_data[numerical_columns])

    # Transform the datasets
    original_processed = preprocess_original_data(original_data, categorical_columns, numerical_columns, one_hot_encoder, scaler)
    synthetic_processed = preprocess_original_data(synthetic_data, categorical_columns, numerical_columns, one_hot_encoder, scaler)

    # Visual Inspection: Plot histograms for each feature
    for feature in numerical_columns:
        plt.figure(figsize=(12, 6))
        sns.histplot(original_processed[:, original_data.columns.get_loc(feature)], color='blue', label='Original', kde=True, stat="density", bins=30)
        sns.histplot(synthetic_processed[:, synthetic_data.columns.get_loc(feature)], color='orange', label='Synthetic', kde=True, stat="density", bins=30)
        plt.title(f'Distribution of {feature}')
        plt.legend()
        plt.show()

    # Statistical Test: Kolmogorov-Smirnov Test
    for feature in numerical_columns:
        stat, p_value = stats.ks_2samp(original_processed[:, original_data.columns.get_loc(feature)], synthetic_processed[:, synthetic_data.columns.get_loc(feature)])
        print(f'K-S test for {feature}: Statistic={stat}, p-value={p_value}')

    # Correlation Comparison: Compute correlation matrices
    original_corr = pd.DataFrame(original_processed).corr()
    synthetic_corr = pd.DataFrame(synthetic_processed).corr()

    plt.figure(figsize=(12, 6))
    sns.heatmap(original_corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap - Original Data')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.heatmap(synthetic_corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap - Synthetic Data')
    plt.show()

def train_neural_network_on_synthetic_data(synthetic_data_file, epochs=5, chunk_size=1000, learning_rate=0.005, activation='elu', sample_size=10000):
    # Create the "weights" directory if it doesn't exist
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

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

    # Prepare file for saving accuracy
    accuracy_file = "accuracy_results.txt"
    
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}")
        
        # Randomly sample `sample_size` entries from the dataset for this epoch
        sampled_data = data.sample(n=sample_size, random_state=epoch).reset_index(drop=True)  # Reset index for sampled data
        
        epoch_accuracies = []
        epoch_losses = []

        for start_row in range(0, len(sampled_data), chunk_size):
            # Select a chunk of sampled data
            end_row = min(start_row + chunk_size, len(sampled_data))
            chunk = sampled_data.iloc[start_row:end_row]

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

        # Save the model weights and accuracy results every 10 epochs
        if (epoch + 1) % 10 == 0:
            weights_path = os.path.join(weights_dir, f"model_weights_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), weights_path)
            with open(accuracy_file, "a") as f:
                f.write(f"Epoch {epoch + 1}: Accuracy: {overall_epoch_accuracy * 100:.2f}% - Loss: {average_loss:.4f}\n")

    return model

def train_neural_network_on_original_data(original_data_file, epochs=5, chunk_size=1000, learning_rate=0.005, activation='elu'):
    """
    Train the Enhanced Neural Network on the original dataset.
    
    Args:
    - original_data_file (str): Path to the original data CSV file.
    - epochs (int): Number of training epochs.
    - chunk_size (int): Size of data chunks for batch processing.
    - learning_rate (float): Learning rate for the optimizer.
    - activation (str): Activation function to use in the model.
    
    Returns:
    - model: Trained Enhanced Neural Network model.
    """
    # Load the original dataset
    data = pd.read_csv(original_data_file)
    
    # Preprocess the original data
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    scaler = StandardScaler()
    
    # Fit encoders on the categorical and numerical columns
    one_hot_encoder.fit(data[categorical_columns])
    scaler.fit(data[numerical_columns])
    
    # Preprocess features and encode the target column
    X_data = preprocess_original_data(data, categorical_columns, numerical_columns, one_hot_encoder, scaler)
    label_encoder = LabelEncoder()
    y_data = label_encoder.fit_transform(data['readmitted'])
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the neural network model
    input_size = X_data.shape[1]
    model = EnhancedNeuralNetwork(input_size=input_size, activation=activation).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-14)
    loss_function = nn.CrossEntropyLoss()

    # Training process
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}")
        
        # Shuffle data at the start of each epoch
        indices = np.arange(len(X_data))
        np.random.shuffle(indices)
        X_data = X_data[indices]
        y_data = y_data[indices]

        epoch_accuracies = []
        epoch_losses = []

        # Train in chunks
        for start_row in range(0, len(X_data), chunk_size):
            end_row = min(start_row + chunk_size, len(X_data))
            X_chunk = X_data[start_row:end_row]
            y_chunk = y_data[start_row:end_row]

            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X_chunk, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_chunk, dtype=torch.long).to(device)

            # Training step
            model.train()
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = loss_function(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            # Evaluate model on current chunk
            model.eval()
            with torch.no_grad():
                predictions = model(X_tensor)
                _, predicted_labels = torch.max(predictions, 1)
                accuracy = accuracy_score(y_tensor.cpu(), predicted_labels.cpu())
                epoch_accuracies.append(accuracy)
                epoch_losses.append(loss.item())

        # Calculate overall accuracy and average loss for the epoch
        overall_epoch_accuracy = np.mean(epoch_accuracies)
        average_loss = np.mean(epoch_losses)
        print(f"Overall Accuracy after Epoch {epoch + 1}: {overall_epoch_accuracy * 100:.2f}% - Average Loss: {average_loss:.4f}")

    return model

def update_model_input_size(model, new_input_size, activation='elu'):
    """
    Update the input size of the model by reinitializing it with the specified new input size.

    Args:
    - model: The existing model instance to be updated.
    - new_input_size (int): The new input size for the model.
    - activation (str): The activation function to use ('elu' by default).

    Returns:
    - model: A new model instance with the updated input size.
    """
    # Create a new instance of the model with the updated input size
    updated_model = EnhancedNeuralNetwork(input_size=new_input_size, activation=activation)
    print(f"Model input size updated to {new_input_size}")
    return updated_model

def evaluate_model(model, weights_file_path, data_file, categorical_columns, numerical_columns, activation='elu'):
    """
    Load model weights and evaluate the model on the specified dataset.
    Produces a confusion matrix, ROC curve, and key performance metrics.

    Args:
    - model: The neural network model instance.
    - weights_file_path (str): Path to the file containing saved model weights.
    - data_file (str): Path to the CSV file containing data for evaluation.
    - categorical_columns (list): List of categorical columns in the data.
    - numerical_columns (list): List of numerical columns in the data.
    - activation (str): Activation function used in the model (default 'elu').

    Returns:
    - metrics (dict): Dictionary containing accuracy, precision, recall, F1-score, and AUC.
    """
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess dataset
    data = pd.read_csv(data_file)
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    scaler = StandardScaler()
    one_hot_encoder.fit(data[categorical_columns])
    scaler.fit(data[numerical_columns])
    X_data = preprocess_original_data(data, categorical_columns, numerical_columns, one_hot_encoder, scaler)

    # Dynamically update model input size and load weights
    model = load_model_with_weights(model, weights_file_path, data_file, categorical_columns, numerical_columns, activation)
    model.to(device)  # Ensure the model is on the correct device

    # Encode target labels and convert to PyTorch tensors
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(data['readmitted'])
    X_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)
    y_true_tensor = torch.tensor(y_true, dtype=torch.long).to(device)

    # Model predictions and probability extraction
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted_labels = torch.max(outputs, 1)
        probabilities = nn.Softmax(dim=1)(outputs)[:, 1]  # Positive class probabilities

    # Convert predictions to numpy arrays and calculate evaluation metrics
    y_pred = predicted_labels.cpu().numpy()
    y_true = y_true_tensor.cpu().numpy()
    probabilities = probabilities.cpu().numpy()

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    print("Classification Report:")
    print(report)

    # Metrics dictionary
    metrics = {
        "Confusion Matrix": conf_matrix,
        "ROC AUC": roc_auc,
        "Classification Report": report
    }

    return metrics

def load_model_with_weights(model, weights_file_path, data_file, categorical_columns, numerical_columns, activation='elu'):
    """
    Load weights into the given model from a specified file path with `weights_only=True`, 
    dynamically updating the input size based on the provided data file.

    Args:
    - model: The neural network model instance with matching architecture.
    - weights_file_path (str): Path to the file containing saved model weights.
    - data_file (str): Path to the CSV file for determining input size.
    - categorical_columns (list): List of categorical columns in the data.
    - numerical_columns (list): List of numerical columns in the data.
    - activation (str): Activation function used in the model (default 'elu').

    Returns:
    - model: Model with loaded weights.
    """
    # Load the data to determine the correct input size
    data = pd.read_csv(data_file)
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    scaler = StandardScaler()
    one_hot_encoder.fit(data[categorical_columns])
    scaler.fit(data[numerical_columns])
    X_data = preprocess_original_data(data, categorical_columns, numerical_columns, one_hot_encoder, scaler)

    # Determine and update the input size
    new_input_size = X_data.shape[1]
    model = update_model_input_size(model, new_input_size, activation=activation)

    # Load weights with weights_only=True
    try:
        state_dict = torch.load(weights_file_path, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()  # Set model to evaluation mode
        print(f"Model weights loaded from '{weights_file_path}' with input size {new_input_size}")
    except RuntimeError as e:
        print("Model loading failed:", e)
        raise

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
# Example usage of evaluate_model
#weights_file_path = 'weights/model_weights_epoch_4000.pth'  # Specify path to weights file
data_file = 'new_data.csv'  # Specify path to data file for evaluation

# Load the original dataset
original_data = pd.read_csv(original_data_file)

# Load the synthetic dataset
synthetic_data = pd.read_csv(synthetic_data_file)

# Compare the datasets
compare_datasets(original_data, synthetic_data)

#Train the model using the original dataset
trained_model = train_neural_network_on_original_data(
    original_data_file,
    epochs=100,
    chunk_size=1000,
    learning_rate=0.005,
    activation='elu'
)   

# # Train the model using the synthetic dataset
# trained_model = train_neural_network_on_synthetic_data(
#     synthetic_data_file,
#     epochs=5,
#     chunk_size=1000,
#     learning_rate=0.005,
#     activation='elu'
# )

# Test the model on the original dataset
test_model_on_original_data(
    model=trained_model,
    original_data_file=original_data_file,
    categorical_columns=categorical_columns,
    numerical_columns=numerical_columns
)

# Initialize the model
input_size = len(categorical_columns) + len(numerical_columns)  # Adjust this based on your input dimensions
model = EnhancedNeuralNetwork(input_size=input_size, activation='elu')

# Load the model with specified weights
#loaded_model = load_model_with_weights(model, weights_file_path, original_data_file, categorical_columns=categorical_columns, numerical_columns=numerical_columns, activation="elu")


# # Evaluate the model
# metrics = evaluate_model(
#     model=loaded_model,
#     weights_file_path=weights_file_path,
#     data_file=original_data_file,
#     categorical_columns=categorical_columns,
#     numerical_columns=numerical_columns
# )







