import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# Define the VAE class
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # Mean and log-variance
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # To normalize output between 0 and 1
        )

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mean, log_var = torch.chunk(encoded, 2, dim=1)
        z = self.reparameterize(mean, log_var)
        decoded = self.decoder(z)
        return decoded, mean, log_var

# Define the VAE loss function
def vae_loss_function(recon_x, x, mean, log_var):
    reconstruction_loss = nn.MSELoss()(recon_x, x)
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reconstruction_loss + kl_divergence

# Function to train the VAE
def train_vae(data, num_epochs=30, batch_size=64, learning_rate=0.001, latent_dim=16):
    input_dim = data.shape[1]
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    dataset = torch.tensor(data, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            recon, mean, log_var = vae(batch)
            loss = vae_loss_function(recon, batch, mean, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
    
    return vae

# Generate synthetic data using the trained VAE
def generate_synthetic_data_with_vae(vae, num_samples=5000):
    vae.eval()
    latent_dim = vae.encoder[-1].out_features // 2
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        synthetic_data = vae.decoder(z).numpy()
    
    return synthetic_data

def preprocess_and_augment_data_with_vae_smote(file_path, output_file_path, num_synthetic_samples=5000):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Define categorical and numerical columns (replace with your actual column names)
    categorical_columns = ['age', 'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'glucose_test', 'A1Ctest', 'change', 'diabetes_med']
    numerical_columns = ['time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']
    target_column = 'readmitted'
    
    # Preprocess categorical variables
    one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_categorical = one_hot_encoder.fit_transform(data[categorical_columns])
    
    # Standardize numerical features
    scaler = StandardScaler()
    numerical_data = scaler.fit_transform(data[numerical_columns])
    
    # Combine numerical and encoded categorical features
    X = np.hstack((numerical_data, encoded_categorical))
    y = LabelEncoder().fit_transform(data[target_column])
    
    # Train the VAE on the combined features
    vae = train_vae(X, num_epochs=30, batch_size=64, learning_rate=0.001, latent_dim=16)
    
    # Generate synthetic data with the trained VAE
    synthetic_data = generate_synthetic_data_with_vae(vae, num_samples=num_synthetic_samples)
    
    # Combine original and synthetic data
    combined_X = np.vstack((X, synthetic_data))
    combined_y = np.hstack((y, np.random.choice(y, size=synthetic_data.shape[0])))  # Randomly assign synthetic labels

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(combined_X, combined_y)

    # Create a new DataFrame from the resampled data
    resampled_df = pd.DataFrame(X_resampled, columns=[*numerical_columns, *one_hot_encoder.get_feature_names_out(categorical_columns)])
    resampled_df[target_column] = y_resampled

    # Save the new dataset to a CSV file
    resampled_df.to_csv(output_file_path, index=False)
    print(f"New dataset saved to {output_file_path}")

# Specify the file paths
input_file = 'hospital_readmissions.csv'
output_file = 'synthetic_augmented_dataset.csv'

# Run the preprocessing and augmentation function with the specified number of synthetic samples
preprocess_and_augment_data_with_vae_smote(input_file, output_file, num_synthetic_samples=10000)