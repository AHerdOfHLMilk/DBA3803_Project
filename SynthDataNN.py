import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Deep Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2)  # Mean and log-variance
        )
        
        # Deep Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # Normalizing output between 0 and 1
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

def train_vae(data, num_epochs=30, batch_size=64, learning_rate=0.001, latent_dim=32, kl_weight=1.0, mse_threshold=0.05, kl_weight_start=10, kl_weight_end=20):
    input_dim = data.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    dataset = torch.tensor(data, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    kl_weight_increment = (kl_weight_end - kl_weight_start) / num_epochs
    kl_weight = kl_weight_start

    vae.train()
    initial_mse = None  # Track initial MSE for relative accuracy
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_divergence = 0  # Accumulate KL divergence for monitoring
        correct_reconstructions = 0  # For "accuracy" approximation
        total_samples = 0
        kl_weight += kl_weight_increment  # Gradually increase KL weight
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mean, log_var = vae(batch)
            
            # Compute the losses
            loss, recon_loss, kl_div = vae_loss_function(recon, batch, mean, log_var, kl_weight)
            loss.backward()
            optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_divergence += kl_div.item()  # Track KL divergence
            total_samples += batch.size(0)

            # Calculate reconstruction MSE per sample and track relative accuracy
            mse = nn.MSELoss(reduction='none')(recon, batch).mean(dim=1)
            if initial_mse is None:
                initial_mse = mse.mean().item()
            correct_reconstructions += (mse < mse_threshold).float().sum().item()
        
        # Epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_recon_loss = total_recon_loss / len(train_loader)
        epoch_kl_divergence = total_kl_divergence / len(train_loader)  # Average KL divergence per batch
        epoch_accuracy = correct_reconstructions / total_samples
        relative_accuracy = (1 - (epoch_recon_loss / initial_mse)) * 100

        # Print epoch metrics, including KL divergence
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Recon Loss: {epoch_recon_loss:.4f}, "
              f"KL Divergence: {epoch_kl_divergence:.4f}, Accuracy: {epoch_accuracy:.2%}, "
              f"Relative Accuracy: {relative_accuracy:.2f}%")

    return vae

def vae_loss_function(recon_x, x, mean, log_var, kl_weight=10.0):
    # Compute the reconstruction loss (e.g., MSE Loss)
    reconstruction_loss = nn.MSELoss()(recon_x, x)
    
    # Compute KL divergence
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    
    # Return the weighted sum of reconstruction loss and KL divergence
    total_loss = reconstruction_loss + kl_weight * kl_divergence
    return total_loss, reconstruction_loss, kl_divergence

def generate_synthetic_data_with_vae(vae, num_samples=5000):
    vae.eval()
    latent_dim = vae.encoder[-1].out_features // 2
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(vae.device)  # Move z to the same device as the VAE
        synthetic_data = vae.decoder(z).cpu().numpy()  # Move the output to CPU before converting to numpy
    
    return synthetic_data

def preprocess_and_augment_data_with_vae(file_path, output_file_path, num_synthetic_samples=5000):
    # Load dataset
    data = pd.read_csv(file_path)
    
    # Define categorical and numerical columns
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

    # Split data by class labels
    X_yes = X[y == 1]  # Data with label 'yes'
    X_no = X[y == 0]   # Data with label 'no'
    
    # Train separate VAE models for each class
    print("Training VAE for 'readmitted = yes'")
    vae_yes = train_vae(X_yes, num_epochs=300, batch_size=64, learning_rate=0.00002, latent_dim=128)
    print("Training VAE for 'readmitted = no'")
    vae_no = train_vae(X_no, num_epochs=300, batch_size=64, learning_rate=0.00002, latent_dim=128)
    
    # Generate synthetic data for each class
    synthetic_yes = generate_synthetic_data_with_vae(vae_yes, num_samples=num_synthetic_samples // 2)
    synthetic_no = generate_synthetic_data_with_vae(vae_no, num_samples=num_synthetic_samples // 2)

    # Assign corresponding labels
    synthetic_labels_yes = np.ones(synthetic_yes.shape[0], dtype=int)  # Label 1 for 'yes'
    synthetic_labels_no = np.zeros(synthetic_no.shape[0], dtype=int)   # Label 0 for 'no'

    # Combine the synthetic data and labels
    synthetic_data = np.vstack((synthetic_yes, synthetic_no))
    synthetic_labels = np.hstack((synthetic_labels_yes, synthetic_labels_no))

    # Shuffle the synthetic data and labels together
    indices = np.arange(synthetic_data.shape[0])
    np.random.shuffle(indices)
    synthetic_data = synthetic_data[indices]
    synthetic_labels = synthetic_labels[indices]

    # Create a new DataFrame from the shuffled synthetic data
    synthetic_df = pd.DataFrame(synthetic_data, columns=[*numerical_columns, *one_hot_encoder.get_feature_names_out(categorical_columns)])
    synthetic_df[target_column] = synthetic_labels

    # Save the new synthetic-only dataset to a CSV file
    synthetic_df.to_csv(output_file_path, index=False)
    print(f"Synthetic-only dataset saved to {output_file_path}")

# Specify the file paths
input_file = 'hospital_readmissions.csv'
output_file = 'synthetic_augmented_dataset.csv'

# Run the preprocessing and augmentation function with the specified number of synthetic samples
preprocess_and_augment_data_with_vae(input_file, output_file, num_synthetic_samples=1000000)