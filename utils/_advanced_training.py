import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import warnings
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
from ._modeltraining import EmotionalLatentDataset, EmotionalVAE, VAETrainer

class AdvancedEmotionalVAE(nn.Module):
    """Advanced VAE with different architectures and regularization"""
    
    def __init__(self, latent_dim, hidden_dims, condition_dim=2, 
                 architecture='standard', use_attention=False, dropout_rate=0.2):
        super(AdvancedEmotionalVAE, self).__init__()
        
        self.latent_dim = latent_dim  # 750
        self.hidden_dims = hidden_dims
        self.condition_dim = condition_dim
        self.quantizer_count = 8
        self.architecture = architecture
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        
        input_dim = self.quantizer_count * self.latent_dim
        
        # Build different architectures
        if architecture == 'standard':
            self._build_standard_architecture(input_dim)
        elif architecture == 'residual':
            self._build_residual_architecture(input_dim)
        elif architecture == 'deep':
            self._build_deep_architecture(input_dim)
        elif architecture == 'hierarchical':
            self._build_hierarchical_architecture(input_dim)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _build_standard_architecture(self, input_dim):
        """Standard VAE architecture with improvements"""
        # Encoder
        encoder_layers = []
        in_channels = input_dim
        
        for i, h_dim in enumerate(self.hidden_dims):
            encoder_layers.extend([
                nn.Linear(in_channels, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(self.dropout_rate)
            ])
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space projection
        self.mu = nn.Linear(self.hidden_dims[-1], input_dim)
        self.var = nn.Linear(self.hidden_dims[-1], input_dim)
        
        # Decoder
        decoder_layers = []
        decoder_input_dim = input_dim + self.condition_dim
        
        for i in reversed(range(len(self.hidden_dims))):
            h_dim = self.hidden_dims[i]
            decoder_layers.extend([
                nn.Linear(decoder_input_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(self.dropout_rate)
            ])
            decoder_input_dim = h_dim
            
        decoder_layers.extend([
            nn.Linear(decoder_input_dim, input_dim),
            nn.Tanh()  # Output activation
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _build_residual_architecture(self, input_dim):
        """Residual connections for deeper networks"""
        self.encoder_blocks = nn.ModuleList()
        in_channels = input_dim
        
        for h_dim in self.hidden_dims:
            block = ResidualBlock(in_channels, h_dim, self.dropout_rate)
            self.encoder_blocks.append(block)
            in_channels = h_dim
        
        self.mu = nn.Linear(self.hidden_dims[-1], input_dim)
        self.var = nn.Linear(self.hidden_dims[-1], input_dim)
        
        self.decoder_blocks = nn.ModuleList()
        decoder_input_dim = input_dim + self.condition_dim
        
        for i in reversed(range(len(self.hidden_dims))):
            h_dim = self.hidden_dims[i]
            block = ResidualBlock(decoder_input_dim, h_dim, self.dropout_rate)
            self.decoder_blocks.append(block)
            decoder_input_dim = h_dim
        
        self.decoder_output = nn.Sequential(
            nn.Linear(decoder_input_dim, input_dim),
            nn.Tanh()
        )
    
    def _build_deep_architecture(self, input_dim):
        """Deeper network with more layers"""
        # Double the hidden dimensions for deeper network
        deep_hidden_dims = []
        for h_dim in self.hidden_dims:
            deep_hidden_dims.extend([h_dim, h_dim // 2])
        
        encoder_layers = []
        in_channels = input_dim
        
        for i, h_dim in enumerate(deep_hidden_dims):
            encoder_layers.extend([
                nn.Linear(in_channels, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ELU(),
                nn.Dropout(self.dropout_rate)
            ])
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.mu = nn.Linear(deep_hidden_dims[-1], input_dim)
        self.var = nn.Linear(deep_hidden_dims[-1], input_dim)
        
        # Decoder
        decoder_layers = []
        decoder_input_dim = input_dim + self.condition_dim
        
        for i in reversed(range(len(deep_hidden_dims))):
            h_dim = deep_hidden_dims[i]
            decoder_layers.extend([
                nn.Linear(decoder_input_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ELU(),
                nn.Dropout(self.dropout_rate)
            ])
            decoder_input_dim = h_dim
            
        decoder_layers.extend([
            nn.Linear(decoder_input_dim, input_dim),
            nn.Tanh()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _build_hierarchical_architecture(self, input_dim):
        """Hierarchical VAE with multiple latent levels"""
        # First level encoder
        self.encoder_l1 = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.LeakyReLU(0.2)
        )
        
        # Second level encoder
        self.encoder_l2 = nn.Sequential(
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.BatchNorm1d(self.hidden_dims[1]),
            nn.LeakyReLU(0.2)
        )
        
        # Latent projections for each level
        self.mu_l1 = nn.Linear(self.hidden_dims[0], self.hidden_dims[0])
        self.var_l1 = nn.Linear(self.hidden_dims[0], self.hidden_dims[0])
        
        self.mu_l2 = nn.Linear(self.hidden_dims[1], input_dim)
        self.var_l2 = nn.Linear(self.hidden_dims[1], input_dim)
        
        # Hierarchical decoder
        self.decoder_l2 = nn.Sequential(
            nn.Linear(input_dim + self.condition_dim, self.hidden_dims[1]),
            nn.BatchNorm1d(self.hidden_dims[1]),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder_l1 = nn.Sequential(
            nn.Linear(self.hidden_dims[1] + self.hidden_dims[0], self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder_output = nn.Sequential(
            nn.Linear(self.hidden_dims[0], input_dim),
            nn.Tanh()
        )
    
    def encode(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        if self.architecture == 'standard' or self.architecture == 'deep':
            h = self.encoder(x_flat)
            return self.mu(h), self.var(h)
        
        elif self.architecture == 'residual':
            h = x_flat
            for block in self.encoder_blocks:
                h = block(h)
            return self.mu(h), self.var(h)
        
        elif self.architecture == 'hierarchical':
            h1 = self.encoder_l1(x_flat)
            h2 = self.encoder_l2(h1)
            
            mu_l1, var_l1 = self.mu_l1(h1), self.var_l1(h1)
            mu_l2, var_l2 = self.mu_l2(h2), self.var_l2(h2)
            
            return (mu_l1, mu_l2), (var_l1, var_l2)
    
    def reparameterize(self, mu, logvar):
        if self.architecture == 'hierarchical':
            mu_l1, mu_l2 = mu
            logvar_l1, logvar_l2 = logvar
            
            std_l1 = torch.exp(0.5 * logvar_l1)
            std_l2 = torch.exp(0.5 * logvar_l2)
            
            eps_l1 = torch.randn_like(std_l1)
            eps_l2 = torch.randn_like(std_l2)
            
            z_l1 = mu_l1 + eps_l1 * std_l1
            z_l2 = mu_l2 + eps_l2 * std_l2
            
            return z_l1, z_l2
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
    
    def decode(self, z, condition):
        if self.architecture == 'hierarchical':
            z_l1, z_l2 = z
            z_cond = torch.cat([z_l2, condition], dim=1)
            
            h2 = self.decoder_l2(z_cond)
            h1_cond = torch.cat([h2, z_l1], dim=1)
            h1 = self.decoder_l1(h1_cond)
            output = self.decoder_output(h1)
        else:
            z_cond = torch.cat([z, condition], dim=1)
            
            if self.architecture == 'residual':
                h = z_cond
                for block in self.decoder_blocks:
                    h = block(h)
                output = self.decoder_output(h)
            else:
                output = self.decoder(z_cond)
        
        return output.view(output.size(0), self.quantizer_count, self.latent_dim)
    
    def forward(self, x, condition):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, condition)
        return x_recon, mu, logvar
    
    def sample(self, num_samples, condition, device):
        if self.architecture == 'hierarchical':
            z_l1 = torch.randn(num_samples, self.hidden_dims[0]).to(device)
            z_l2 = torch.randn(num_samples, self.quantizer_count * self.latent_dim).to(device)
            z = (z_l1, z_l2)
        else:
            z = torch.randn(num_samples, self.quantizer_count * self.latent_dim).to(device)
        
        if len(condition.shape) == 1:
            condition = condition.unsqueeze(0).repeat(num_samples, 1)
        
        samples = self.decode(z, condition)
        return samples

class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    def __init__(self, in_dim, out_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        return self.activation(self.main(x) + self.shortcut(x))

class LossCalculator:
    """Different loss functions for VAE training"""
    
    @staticmethod
    def beta_vae_loss(recon_x, x, mu, logvar, beta=1.0):
        """Standard Beta-VAE loss"""
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    @staticmethod
    def cyclical_annealing_loss(recon_x, x, mu, logvar, epoch, cycle_length=10, n_cycles=4):
        """Cyclical annealing of KL term"""
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Cyclical annealing
        cycle = (epoch % cycle_length) / cycle_length
        beta = min(1.0, cycle * n_cycles / cycle_length)
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    @staticmethod
    def annealed_vae_loss(recon_x, x, mu, logvar, epoch, max_epochs=100):
        """Linearly annealed KL term"""
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Linear annealing
        beta = min(1.0, epoch / (max_epochs * 0.5))
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    @staticmethod
    def emotion_weighted_loss(recon_x, x, mu, logvar, conditions, beta=1.0):
        """Emotion-aware weighted loss"""
        recon_loss = F.mse_loss(recon_x, x, reduction='none').mean(dim=[1, 2])
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        # Weight based on emotional intensity
        valence, arousal = conditions[:, 0], conditions[:, 1]
        emotion_intensity = torch.sqrt(valence**2 + arousal**2)
        weights = 1.0 + emotion_intensity  # Higher weight for more intense emotions
        
        weighted_recon = (recon_loss * weights).mean()
        weighted_kl = (kl_loss * weights).mean()
        
        return weighted_recon + beta * weighted_kl, weighted_recon, weighted_kl
    
    @staticmethod
    def hierarchical_loss(recon_x, x, mu, logvar, beta=1.0):
        """Loss for hierarchical VAE"""
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        if isinstance(mu, tuple):
            mu_l1, mu_l2 = mu
            logvar_l1, logvar_l2 = logvar
            
            kl_l1 = -0.5 * torch.mean(1 + logvar_l1 - mu_l1.pow(2) - logvar_l1.exp())
            kl_l2 = -0.5 * torch.mean(1 + logvar_l2 - mu_l2.pow(2) - logvar_l2.exp())
            
            kl_loss = kl_l1 + kl_l2
        else:
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

class AdvancedVAETrainer:
    """Advanced trainer with multiple configurations and fine-tuning"""
    
    def __init__(
        self,
        model_config,
        training_config,
        latent_dir='latent_representations',
        output_dir='advanced_vae_models',
        device=None
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.latent_dir = latent_dir
        self.output_dir = output_dir
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Create output directory
        model_name = f"{model_config['architecture']}_{training_config['loss_type']}"
        self.model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(self.model_output_dir, exist_ok=True)
        
        # Initialize model
        self.model = AdvancedEmotionalVAE(**model_config)
        self.model.to(self.device)
        
        # Initialize optimizer
        if training_config['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=training_config['lr'],
                betas=(0.9, 0.999),
                weight_decay=training_config.get('weight_decay', 0)
            )
        elif training_config['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=training_config['lr'],
                weight_decay=training_config.get('weight_decay', 0.01)
            )
        elif training_config['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=training_config['lr'],
                momentum=training_config.get('momentum', 0.9)
            )
        
        # Learning rate scheduler
        if training_config.get('use_scheduler', False):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=10, factor=0.5
            )
        else:
            self.scheduler = None
        
        # Loss calculator
        self.loss_calculator = LossCalculator()
        
    def compute_loss(self, recon_x, x, mu, logvar, conditions, epoch=0):
        """Compute loss based on configuration"""
        loss_type = self.training_config['loss_type']
        beta = self.training_config.get('beta', 1.0)
        
        if loss_type == 'beta_vae':
            return self.loss_calculator.beta_vae_loss(recon_x, x, mu, logvar, beta)
        elif loss_type == 'cyclical':
            return self.loss_calculator.cyclical_annealing_loss(recon_x, x, mu, logvar, epoch)
        elif loss_type == 'annealed':
            max_epochs = self.training_config['epochs']
            return self.loss_calculator.annealed_vae_loss(recon_x, x, mu, logvar, epoch, max_epochs)
        elif loss_type == 'emotion_weighted':
            return self.loss_calculator.emotion_weighted_loss(recon_x, x, mu, logvar, conditions, beta)
        elif loss_type == 'hierarchical':
            return self.loss_calculator.hierarchical_loss(recon_x, x, mu, logvar, beta)
        else:
            return self.loss_calculator.beta_vae_loss(recon_x, x, mu, logvar, beta)
    
    def train(self, max_samples=None):
        """Train the model"""
        # Prepare dataset
        dataset = EmotionalLatentDataset(
            self.latent_dir, 
            self.training_config['chunk_size'], 
            max_samples
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.training_config['batch_size'], 
            shuffle=True,
            num_workers=4
        )
        
        # Training statistics
        stats = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'learning_rate': []
        }
        
        best_loss = float('inf')
        patience_counter = 0
        max_patience = self.training_config.get('early_stopping_patience', 20)
        
        # Training loop
        for epoch in range(self.training_config['epochs']):
            self.model.train()
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.training_config['epochs']}")
            
            for batch in progress_bar:
                latents = batch['latent'].to(self.device)
                conditions = torch.stack([batch['valence'], batch['arousal']], dim=1).to(self.device)
                
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(latents, conditions)
                
                loss, recon_loss, kl_loss = self.compute_loss(
                    recon_batch, latents, mu, logvar, conditions, epoch
                )
                
                loss.backward()
                
                # Gradient clipping
                if self.training_config.get('grad_clip', False):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'recon': f"{recon_loss.item():.4f}",
                    'kl': f"{kl_loss.item():.4f}"
                })
            
            # Compute averages
            avg_loss = epoch_loss / len(dataloader)
            avg_recon = epoch_recon_loss / len(dataloader)
            avg_kl = epoch_kl_loss / len(dataloader)
            
            stats['total_loss'].append(avg_loss)
            stats['recon_loss'].append(avg_recon)
            stats['kl_loss'].append(avg_kl)
            stats['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.save_model("best_model.pt")
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Periodic saves and visualizations
            if (epoch + 1) % 10 == 0:
                self.save_model(f"model_epoch_{epoch+1}.pt")
                self.visualize_training_progress(stats, epoch + 1)
        
        # Save final model and statistics
        self.save_model("final_model.pt")
        self.save_training_stats(stats)
        self.create_final_report(stats)
        
        return stats
    
    def fine_tune(self, pretrained_model_path, fine_tune_config, max_samples=None):
        """Fine-tune a pre-trained model"""
        print(f"Loading pre-trained model from {pretrained_model_path}")
        
        # Load pre-trained model
        checkpoint = torch.load(pretrained_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze specific layers if specified
        if fine_tune_config.get('freeze_encoder', False):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            print("Encoder layers frozen")
        
        if fine_tune_config.get('freeze_decoder', False):
            for param in self.model.decoder.parameters():
                param.requires_grad = False
            print("Decoder layers frozen")
        
        # Use lower learning rate for fine-tuning
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = fine_tune_config.get('lr', self.training_config['lr'] * 0.1)
        
        # Update training config for fine-tuning
        original_epochs = self.training_config['epochs']
        self.training_config['epochs'] = fine_tune_config.get('epochs', original_epochs // 2)
        
        print(f"Fine-tuning for {self.training_config['epochs']} epochs with lr={param_group['lr']}")
        
        # Fine-tune
        stats = self.train(max_samples)
        
        # Restore original config
        self.training_config['epochs'] = original_epochs
        
        return stats
    
    def save_model(self, filename):
        """Save model with configuration"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': self.model_config,
            'training_config': self.training_config
        }, os.path.join(self.model_output_dir, filename))
    
    def save_training_stats(self, stats):
        """Save training statistics"""
        with open(os.path.join(self.model_output_dir, 'training_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
    
    def visualize_training_progress(self, stats, epoch):
        """Visualize training progress"""
        epochs = range(1, len(stats['total_loss']) + 1)
        
        plt.figure(figsize=(15, 10))
        
        # Loss curves
        plt.subplot(2, 3, 1)
        plt.plot(epochs, stats['total_loss'], 'b-', label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 3, 2)
        plt.plot(epochs, stats['recon_loss'], 'g-', label='Reconstruction')
        plt.plot(epochs, stats['kl_loss'], 'r-', label='KL Divergence')
        plt.title('Loss Components')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate
        plt.subplot(2, 3, 3)
        plt.plot(epochs, stats['learning_rate'], 'm-', label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('LR')
        plt.legend()
        plt.grid(True)
        
        # Loss ratio
        plt.subplot(2, 3, 4)
        ratio = np.array(stats['kl_loss']) / np.array(stats['recon_loss'])
        plt.plot(epochs, ratio, 'orange', label='KL/Recon Ratio')
        plt.title('KL/Reconstruction Ratio')
        plt.xlabel('Epochs')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True)
        
        # Moving averages
        plt.subplot(2, 3, 5)
        window = min(10, len(stats['total_loss']))
        if len(stats['total_loss']) >= window:
            moving_avg = pd.Series(stats['total_loss']).rolling(window=window).mean()
            plt.plot(epochs, moving_avg, 'purple', label=f'MA({window})')
            plt.title('Moving Average Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
        
        # Recent performance
        plt.subplot(2, 3, 6)
        recent_epochs = max(1, len(epochs) - 20)
        plt.plot(epochs[recent_epochs:], stats['total_loss'][recent_epochs:], 'b-')
        plt.title('Recent Loss (Last 20 epochs)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_output_dir, f'training_progress_epoch_{epoch}.png'))
        plt.close()
    
    def create_final_report(self, stats):
        """Create a comprehensive training report"""
        report = {
            'model_config': self.model_config,
            'training_config': self.training_config,
            'final_metrics': {
                'final_total_loss': stats['total_loss'][-1],
                'final_recon_loss': stats['recon_loss'][-1],
                'final_kl_loss': stats['kl_loss'][-1],
                'best_total_loss': min(stats['total_loss']),
                'epochs_trained': len(stats['total_loss'])
            },
            'training_summary': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'convergence_epoch': np.argmin(stats['total_loss']) + 1,
                'final_learning_rate': stats['learning_rate'][-1] if stats['learning_rate'] else None
            }
        }
        
        with open(os.path.join(self.model_output_dir, 'training_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Training completed. Final loss: {stats['total_loss'][-1]:.4f}")
        print(f"Best loss: {min(stats['total_loss']):.4f} at epoch {np.argmin(stats['total_loss']) + 1}")
        print(f"Model saved in: {self.model_output_dir}")

class ModelComparator:
    """Compare different model configurations"""
    
    def __init__(self, results_dir='advanced_vae_models'):
        self.results_dir = results_dir
        self.model_results = {}
    
    def load_all_results(self):
        """Load all training results"""
        for model_dir in os.listdir(self.results_dir):
            model_path = os.path.join(self.results_dir, model_dir)
            if os.path.isdir(model_path):
                stats_path = os.path.join(model_path, 'training_stats.json')
                report_path = os.path.join(model_path, 'training_report.json')
                
                if os.path.exists(stats_path) and os.path.exists(report_path):
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    
                    self.model_results[model_dir] = {
                        'stats': stats,
                        'report': report
                    }
    
    def create_comparison_report(self):
        """Create a comprehensive comparison report"""
        if not self.model_results:
            self.load_all_results()
        
        comparison_data = []
        for model_name, data in self.model_results.items():
            report = data['report']
            stats = data['stats']
            
            comparison_data.append({
                'Model': model_name,
                'Architecture': report['model_config']['architecture'],
                'Loss Type': report['training_config']['loss_type'],
                'Best Loss': report['final_metrics']['best_total_loss'],
                'Final Loss': report['final_metrics']['final_total_loss'],
                'Convergence Epoch': report['training_summary']['convergence_epoch'],
                'Total Parameters': report['training_summary']['total_parameters'],
                'Epochs Trained': report['final_metrics']['epochs_trained']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Best Loss')
        
        # Save comparison table
        df.to_csv(os.path.join(self.results_dir, 'model_comparison.csv'), index=False)
        
        # Create visualization
        self.visualize_comparison(df)
        
        return df
    
    def visualize_comparison(self, df):
        """Visualize model comparison"""
        plt.figure(figsize=(20, 12))
        
        # Best loss comparison
        plt.subplot(2, 3, 1)
        sns.barplot(data=df, x='Model', y='Best Loss')
        plt.title('Best Loss Comparison')
        plt.xticks(rotation=45)
        
        # Architecture comparison
        plt.subplot(2, 3, 2)
        arch_counts = df['Architecture'].value_counts()
        plt.pie(arch_counts.values, labels=arch_counts.index, autopct='%1.1f%%')
        plt.title('Architecture Distribution')
        
        # Loss type comparison
        plt.subplot(2, 3, 3)
        loss_counts = df['Loss Type'].value_counts()
        plt.pie(loss_counts.values, labels=loss_counts.index, autopct='%1.1f%%')
        plt.title('Loss Type Distribution')
        
        # Parameter count vs performance
        plt.subplot(2, 3, 4)
        plt.scatter(df['Total Parameters'], df['Best Loss'])
        plt.xlabel('Total Parameters')
        plt.ylabel('Best Loss')
        plt.title('Parameters vs Performance')
        
        # Convergence speed
        plt.subplot(2, 3, 5)
        sns.barplot(data=df, x='Model', y='Convergence Epoch')
        plt.title('Convergence Speed')
        plt.xticks(rotation=45)
        
        # Training efficiency
        plt.subplot(2, 3, 6)
        efficiency = df['Best Loss'] / df['Epochs Trained']
        plt.bar(df['Model'], efficiency)
        plt.title('Training Efficiency (Loss/Epoch)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return df

# Configuration templates for different model types
CONFIG_TEMPLATES = {
    'standard_beta': {
        'model_config': {
            'latent_dim': 750,
            'hidden_dims': [512, 256, 128],
            'condition_dim': 2,
            'architecture': 'standard',
            'dropout_rate': 0.2
        },
        'training_config': {
            'chunk_size': 10,
            'batch_size': 64,
            'epochs': 100,
            'lr': 1e-4,
            'optimizer': 'adam',
            'loss_type': 'beta_vae',
            'beta': 0.5,
            'use_scheduler': True,
            'grad_clip': True,
            'early_stopping_patience': 20
        }
    },
    'deep_annealed': {
        'model_config': {
            'latent_dim': 750,
            'hidden_dims': [512, 256, 128],
            'condition_dim': 2,
            'architecture': 'deep',
            'dropout_rate': 0.3
        },
        'training_config': {
            'chunk_size': 10,
            'batch_size': 32,
            'epochs': 150,
            'lr': 5e-5,
            'optimizer': 'adamw',
            'loss_type': 'annealed',
            'weight_decay': 0.01,
            'use_scheduler': True,
            'grad_clip': True,
            'early_stopping_patience': 25
        }
    },
    'residual_cyclical': {
        'model_config': {
            'latent_dim': 750,
            'hidden_dims': [768, 384, 192],
            'condition_dim': 2,
            'architecture': 'residual',
            'dropout_rate': 0.1
        },
        'training_config': {
            'chunk_size': 10,
            'batch_size': 48,
            'epochs': 120,
            'lr': 2e-4,
            'optimizer': 'adam',
            'loss_type': 'cyclical',
            'use_scheduler': False,
            'grad_clip': True,
            'early_stopping_patience': 30
        }
    },
    'hierarchical_emotion': {
        'model_config': {
            'latent_dim': 750,
            'hidden_dims': [512, 256],
            'condition_dim': 2,
            'architecture': 'hierarchical',
            'dropout_rate': 0.25
        },
        'training_config': {
            'chunk_size': 10,
            'batch_size': 64,
            'epochs': 100,
            'lr': 1e-4,
            'optimizer': 'adam',
            'loss_type': 'emotion_weighted',
            'beta': 0.8,
            'use_scheduler': True,
            'grad_clip': False,
            'early_stopping_patience': 15
        }
    }
}

def create_training_pipeline(configs, latent_dir='latent_representations', 
                           output_dir='advanced_vae_models', max_samples=None):
    """Create and run multiple training pipelines"""
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n{'='*50}")
        print(f"Training {config_name}")
        print(f"{'='*50}")
        
        trainer = AdvancedVAETrainer(
            model_config=config['model_config'],
            training_config=config['training_config'],
            latent_dir=latent_dir,
            output_dir=output_dir
        )
        
        stats = trainer.train(max_samples=max_samples)
        results[config_name] = stats
        
        print(f"Completed {config_name}")
    
    # Create comparison report
    comparator = ModelComparator(output_dir)
    comparison_df = comparator.create_comparison_report()
    
    print(f"\n{'='*50}")
    print("Training Pipeline Completed")
    print(f"{'='*50}")
    print(f"Results saved in: {output_dir}")
    print("Best performing models:")
    print(comparison_df.head(3)[['Model', 'Architecture', 'Loss Type', 'Best Loss']])
    
    return results, comparison_df
