import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  # Notebook-friendly tqdm
import warnings
import soundfile as sf

class EmotionalLatentDataset(Dataset):
    """Dataset class containing latent representations and emotional labels"""
    
    def __init__(self, latent_dir, chunk_size=10, max_samples=None):
        self.latent_dir = latent_dir
        self.chunk_size = chunk_size
        
        # Load metadata file
        metadata_path = os.path.join(latent_dir, f'metadata_{chunk_size}s.csv')
        self.metadata = pd.read_csv(metadata_path)
        
        # Limit the samples if needed
        if max_samples is not None:
            self.metadata = self.metadata.sample(min(max_samples, len(self.metadata)))
            
        print(f"Loaded a total of {len(self.metadata)} latent representations.")
        
        # Load latent vectors
        self.latents = []
        self.load_latent_vectors()
    
    def load_latent_vectors(self):
        """Loads all latent vectors"""
        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc="Loading latent representations"):
            latent_path = os.path.join(self.latent_dir, f'latents_{self.chunk_size}s', f"{row['song_id']}_{row['chunk_id']}.npy")
            if os.path.exists(latent_path):
                latent = np.load(latent_path)
                # Check the correct shape and adjust if necessary
                if len(latent.shape) != 2:
                    print(f"Warning: Unexpected latent shape {latent.shape}, adjusting...")
                    latent = latent.reshape(8, 750)  # Expected shape for EnCodec latents
                self.latents.append(latent)
            else:
                print(f"Warning: {latent_path} not found!")
                
        # Clean metadata for missing files
        if len(self.latents) != len(self.metadata):
            print(f"Warning: {len(self.metadata) - len(self.latents)} latent files not found, cleaning metadata.")
            self.metadata = self.metadata.iloc[:len(self.latents)]
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        # Return latent vector and emotional labels
        latent = torch.tensor(self.latents[idx], dtype=torch.float32)
        valence = torch.tensor(self.metadata.iloc[idx]['valence_norm'], dtype=torch.float32)
        arousal = torch.tensor(self.metadata.iloc[idx]['arousal_norm'], dtype=torch.float32)
        
        return {
            'latent': latent,  # Shape: [8, 750]
            'valence': valence,
            'arousal': arousal,
            'song_id': self.metadata.iloc[idx]['song_id'],
            'chunk_id': self.metadata.iloc[idx]['chunk_id']
        }

class EncodecTokenHead(nn.Module):
    """EnCodec formatına uygun token çıktıları üreten özel çıkış katmanı"""
    def __init__(self, input_dim, token_dim=750, token_range=256, quantizer_count=8):
        super().__init__()
        self.projection = nn.Linear(input_dim, token_dim * quantizer_count)
        self.token_dim = token_dim
        self.quantizer_count = quantizer_count
        self.token_range = token_range
        
    def forward(self, x):
        projected = self.projection(x)
        # Clamp projeksiyon çıktılarını makul bir aralığa getir
        projected = torch.clamp(projected, -10, 10)
        # Sigmoid ile 0-1 arasına sınırlandırma
        normalized = torch.sigmoid(projected)
        # Ölçeklendirme (0-255 aralığında değerler)
        scaled = normalized * (self.token_range - 1)
        # Yeniden şekillendirme: [batch_size, quantizer_count, token_dim]
        return scaled.view(x.size(0), self.quantizer_count, self.token_dim)

class EmotionalVAE(nn.Module):
    """Variational Autoencoder for emotional music generation"""
    
    def __init__(self, latent_dim, hidden_dims, condition_dim=2, use_encodec_format=True):
        super(EmotionalVAE, self).__init__()
        
        self.latent_dim = latent_dim  # This is the code length (750)
        self.hidden_dims = hidden_dims
        self.condition_dim = condition_dim  # Valence and Arousal values
        self.quantizer_count = 8  # Number of quantizers in EnCodec
        self.use_encodec_format = use_encodec_format
        
        # Input dimension: flattened latent vector of each quantizer (8 x 750)
        input_dim = self.quantizer_count * self.latent_dim
        
        # Encoder modules
        encoder_layers = []
        in_channels = input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space projections
        self.mu = nn.Linear(hidden_dims[-1], self.quantizer_count * self.latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], self.quantizer_count * self.latent_dim)
        
        # Decoder modules
        decoder_layers = []
        
        # Input size for decoder: latent space + condition dimensions
        decoder_input_size = self.quantizer_count * self.latent_dim + condition_dim
        reversed_hidden_dims = list(reversed(hidden_dims))
        
        for i in range(len(reversed_hidden_dims) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(
                        decoder_input_size if i == 0 else reversed_hidden_dims[i],
                        reversed_hidden_dims[i+1]
                    ),
                    nn.LayerNorm(reversed_hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )
        
        # Final decoder layer
        if use_encodec_format:
            # Use special token head for EnCodec compatibility
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(reversed_hidden_dims[-1], reversed_hidden_dims[-1]),
                    nn.LayerNorm(reversed_hidden_dims[-1]),
                    nn.LeakyReLU(),
                    EncodecTokenHead(reversed_hidden_dims[-1], latent_dim, 256, self.quantizer_count)
                )
            )
        else:
            # Standard output layer
            decoder_layers.append(
                nn.Linear(reversed_hidden_dims[-1], input_dim)
            )
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        # Flatten the input: [batch_size, 8, 750] -> [batch_size, 8*750]
        x_flat = x.view(x.size(0), -1)
        
        # Encode
        h = self.encoder(x_flat)
        
        # Get latent parameters
        mu = self.mu(h)
        logvar = self.log_var(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, condition):
        # z shape: [batch_size, 8*750]
        # condition shape: [batch_size, 2]
        
        # Concatenate latent vector and emotional condition
        z_cond = torch.cat([z, condition], dim=1)
        
        # Decoder output
        output = self.decoder(z_cond)
        
        # Reshape to original 2D shape: [batch_size, 8, 750]
        if self.use_encodec_format:
            # Already produces the right format for each quantizer layer
            output_reshaped = output.view(output.size(0), self.quantizer_count, self.latent_dim)
        else:
            output_reshaped = output.view(output.size(0), self.quantizer_count, self.latent_dim)
        
        return output_reshaped
    
    def forward(self, x, condition):
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z, condition)
        
        return x_recon, mu, logvar
    
    def sample(self, num_samples, condition, device):
        """Generates new samples for the given emotional conditions"""
        # Sample random latent vectors
        z = torch.randn(num_samples, self.quantizer_count * self.latent_dim).to(device)
        
        # Expand given conditions
        if len(condition.shape) == 1:
            condition = condition.unsqueeze(0).repeat(num_samples, 1)
        
        # Conditional sampling
        samples = self.decode(z, condition)
        return samples
    
    def reset_parameters(self):
        """Parametreleri daha güvenli değerlerle yeniden ilklendir"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class VAETrainer:
    """Helper class to train the emotional VAE model"""
    
    def __init__(
        self, 
        model, 
        latent_dir='latent_representations',
        output_dir='vae_model',
        chunk_size=10,
        batch_size=64,
        lr=1e-4,
        beta=0.5,  # KL loss weight
        device=None
    ):
        self.model = model
        self.latent_dir = latent_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.beta = beta
        
        # Check for CUDA
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Move model to device
        self.model.to(self.device)
        
    # VAETrainer sınıfında loss_function metodunu güncelleyelim
    def loss_function(self, recon_x, x, mu, logvar):
        """VAE loss function: Reconstruction + KL Divergence + Token Range Loss"""
        # MSE kaybını 'mean' olarak değiştirin ve skala faktörü ekleyin
        recon_loss = F.mse_loss(recon_x, x, reduction='mean') * 0.01  # Ölçeklendirme faktörü
        
        # KL Divergence - mean ile değiştirin
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # EnCodec token format uyumluluk kaybı
        if hasattr(self.model, 'use_encodec_format') and self.model.use_encodec_format:
            # Çıktılar 0-255 aralığında olacak şekilde teşvik et
            token_range_loss = torch.mean(F.relu(torch.abs(recon_x) - 255))
            # Geçişlerin yumuşak olmasını teşvik et
            smoothness_loss = torch.mean(torch.abs(recon_x[:, :, 1:] - recon_x[:, :, :-1]))
            
            total_loss = recon_loss + self.beta * kl_loss + 0.1 * token_range_loss + 0.05 * smoothness_loss
            
            return {
                'loss': total_loss,
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'token_range_loss': token_range_loss.item(),
                'smoothness_loss': smoothness_loss.item()
            }
        else:
            total_loss = recon_loss + self.beta * kl_loss
            
            return {
                'loss': total_loss,
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item()
            }
    
    def train(self, num_epochs=100, max_samples=None):
        """Trains the model"""
        # Prepare Dataset and DataLoader
        dataset = EmotionalLatentDataset(self.latent_dir, self.chunk_size, max_samples)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training statistics
        stats = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                # Move batch data to device
                latents = batch['latent'].to(self.device)
                conditions = torch.stack([batch['valence'], batch['arousal']], dim=1).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(latents, conditions)
                
                # Compute loss
                loss_dict = self.loss_function(recon_batch, latents, mu, logvar)
                loss = loss_dict['loss']
                recon_loss = loss_dict['recon_loss']
                kl_loss = loss_dict['kl_loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping ekleyin
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss
                epoch_kl_loss += kl_loss
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'recon': f"{recon_loss:.4f}",
                    'kl': f"{kl_loss:.4f}"
                })
            
            # Compute average losses at epoch end
            avg_loss = epoch_loss / len(dataloader)
            avg_recon = epoch_recon_loss / len(dataloader)
            avg_kl = epoch_kl_loss / len(dataloader)
            
            # Save statistics
            stats['total_loss'].append(avg_loss)
            stats['recon_loss'].append(avg_recon)
            stats['kl_loss'].append(avg_kl)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Reconstruction: {avg_recon:.4f}, KL: {avg_kl:.4f}")
            
            # Save model every 10 epochs, and visualize
            if (epoch + 1) % 10 == 0:
                self.save_model(f"vae_epoch_{epoch+1}.pt")
                self.visualize_reconstructions(dataloader)
                self.visualize_latent_space(dataloader)
                self.plot_training_curves(stats)
                
        # Save final model
        self.save_model("vae_final.pt")
        
        # Visualize training statistics
        self.plot_training_curves(stats)
        
        return stats
    
    def save_model(self, filename):
        """Saves the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.output_dir, filename))
        
    def load_model(self, filename):
        """Loads a saved model"""
        checkpoint = torch.load(os.path.join(self.output_dir, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def visualize_reconstructions(self, dataloader):
        """Visualizes original and reconstructed examples"""
        self.model.eval()
        
        # Get a batch for testing
        with torch.no_grad():
            batch = next(iter(dataloader))
            latents = batch['latent'].to(self.device)
            conditions = torch.stack([batch['valence'], batch['arousal']], dim=1).to(self.device)
            
            # Visualize first 8 examples
            n_samples = min(8, len(latents))
            
            # Reconstruction
            recon_batch, _, _ = self.model(latents[:n_samples], conditions[:n_samples])
            
            # Move to CPU and convert to numpy
            latents_np = latents[:n_samples].cpu().numpy()
            recon_np = recon_batch.cpu().numpy()
            
            # Visualization
            plt.figure(figsize=(20, 4))
            for i in range(n_samples):
                # Original
                plt.subplot(2, n_samples, i + 1)
                plt.plot(latents_np[i])
                plt.title(f"V: {batch['valence'][i]:.2f}, A: {batch['arousal'][i]:.2f}")
                plt.axis('off')
                
                # Reconstruction
                plt.subplot(2, n_samples, i + 1 + n_samples)
                plt.plot(recon_np[i])
                plt.axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "reconstructions.png"))
            plt.close()
    
    def visualize_latent_space(self, dataloader, n_samples=1000):
        """Visualizes the latent space"""
        self.model.eval()
        
        # Collect encoder outputs
        encoded_samples = []
        valence_values = []
        arousal_values = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i * self.batch_size >= n_samples:
                    break
                
                latents = batch['latent'].to(self.device)
                conditions = torch.stack([batch['valence'], batch['arousal']], dim=1).to(self.device)
                
                # Get encoder outputs
                mu, _ = self.model.encode(latents)
                
                encoded_samples.append(mu.cpu().numpy())
                valence_values.append(batch['valence'].numpy())
                arousal_values.append(batch['arousal'].numpy())
        
        # Concatenate data
        encoded_samples = np.vstack(encoded_samples)
        valence_values = np.concatenate(valence_values)
        arousal_values = np.concatenate(arousal_values)
        
        # Dimensionality reduction with t-SNE
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        latent_tsne = tsne.fit_transform(encoded_samples[:n_samples])
        
        # Colors based on average normalized valence-arousal values
        color_values = 0.5 * (valence_values[:n_samples] + arousal_values[:n_samples])
        
        # Visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            latent_tsne[:, 0], 
            latent_tsne[:, 1],
            c=color_values,
            cmap='coolwarm',
            alpha=0.7,
            s=40
        )
        plt.colorbar(scatter, label='V-A Mean')
        plt.title('Latent Space (t-SNE)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "latent_space_tsne.png"))
        plt.close()
        
    def plot_training_curves(self, stats):
        """Plots training statistics"""
        epochs = range(1, len(stats['total_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Total loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, stats['total_loss'], 'b-', label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Reconstruction and KL loss components
        plt.subplot(1, 2, 2)
        plt.plot(epochs, stats['recon_loss'], 'g-', label='Reconstruction')
        plt.plot(epochs, stats['kl_loss'], 'r-', label='KL Divergence')
        plt.title('Loss Components')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_curves.png"))
        plt.close()
    
    def generate_samples(self, num_samples=5, conditions=None):
        """Generates samples for specified emotional conditions"""
        self.model.eval()
        
        if conditions is None:
            # Create various emotional conditions for testing
            conditions = [
                [1.0, 1.0],  # High valence, high arousal (joyful)
                [0.0, 1.0],  # Low valence, high arousal (angry)
                [0.0, 0.0],  # Low valence, low arousal (sad)
                [1.0, 0.0],  # High valence, low arousal (calm)
                [0.5, 0.5],  # Neutral midpoint
            ]
        
        # Produce num_samples examples for each condition
        all_samples = []
        all_conditions = []
        
        with torch.no_grad():
            for condition in conditions:
                # Convert condition to tensor
                condition_tensor = torch.tensor(condition, dtype=torch.float32).to(self.device)
                
                # Generate samples
                samples = self.model.sample(num_samples, condition_tensor, self.device)
                
                all_samples.append(samples.cpu().numpy())
                all_conditions.extend([condition] * num_samples)
        
        # Convert to NumPy arrays
        all_samples = np.vstack(all_samples)
        all_conditions = np.array(all_conditions)
        
        # Save generated samples
        np.save(os.path.join(self.output_dir, "generated_samples.npy"), all_samples)
        np.save(os.path.join(self.output_dir, "generated_conditions.npy"), all_conditions)
        
        return all_samples, all_conditions
    
    def validate_encodec_compatibility(self, dataloader, lat_gen):
        """EnCodec ile model uyumluluğunu doğrula"""
        self.model.eval()
        test_sample = next(iter(dataloader))
        
        # Bir örnek al
        latent, condition = test_sample
        latent = latent.to(self.device)
        condition = condition.to(self.device)
        
        with torch.no_grad():
            # Model üzerinden rekonstrüksiyon
            recon, _, _ = self.model(latent, condition)
            
            # EnCodec ile ses oluşturma deneyi
            scale = torch.ones(1, 1).to(self.device)
            for i in range(min(3, recon.size(0))):
                sample = recon[i].unsqueeze(0)
                decoded_audio = lat_gen.encodec_model.decode([(sample.long(), scale)])[0]
                
                # Ses dosyasını kaydet
                audio_np = decoded_audio.cpu().numpy()[0]
                output_path = os.path.join(self.output_dir, f"validation_sample_{i}.wav")
                sample_rate = lat_gen.encodec_model.sample_rate
                sf.write(output_path, audio_np, sample_rate)
                
        # İstatistikleri geri döndür
        return {
            "token_min": recon.min().item(),
            "token_max": recon.max().item(),
            "token_mean": recon.mean().item(),
            "token_std": recon.std().item(),
            "validation_samples": min(3, recon.size(0))
        }