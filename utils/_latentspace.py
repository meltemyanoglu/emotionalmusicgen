import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
import librosa
import soundfile as sf
from tqdm.notebook import tqdm  # tqdm for notebooks
from torch.utils.data import Dataset, DataLoader
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
from sklearn.manifold import TSNE
import seaborn as sns


class AudioSegmentDataset(Dataset):
    """Dataset class containing audio segments and associated V-A (Valence-Arousal) data."""
    
    def __init__(self, processed_data_dir, chunk_size=5, max_samples=None):
        self.chunk_size = chunk_size
        self.audio_dir = os.path.join(processed_data_dir, f'chunks_{chunk_size}s')
        self.metadata_dir = os.path.join(processed_data_dir, f'metadata_{chunk_size}s')
        
        # Get the list of audio files
        self.audio_files = sorted([f for f in os.listdir(self.audio_dir) if f.endswith('.wav')])
        
        # Limit the samples if specified (for memory management)
        if max_samples is not None:
            self.audio_files = self.audio_files[:max_samples]
        
        print(f"Found a total of {len(self.audio_files)} segments ({chunk_size}s)")
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        metadata_file = audio_file.replace('.wav', '.json')
        
        # Load audio data and metadata
        audio_path = os.path.join(self.audio_dir, audio_file)
        metadata_path = os.path.join(self.metadata_dir, metadata_file)
        
        # Load the audio file
        y, sr = torchaudio.load(audio_path)
        
        # Load the metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Return the audio tensor and V-A values along with metadata
        return {
            'audio': y,
            'sample_rate': sr,
            'song_id': metadata['song_id'],
            'chunk_id': metadata['chunk_id'],
            'valence': metadata['valence'],
            'arousal': metadata['arousal'],
            'valence_norm': metadata['valence_normalized'],
            'arousal_norm': metadata['arousal_normalized'],
            'file_name': audio_file
        }


class LatentRepresentationGenerator:
    """Generates latent representations for audio segments using the EnCodec model."""
    
    def __init__(
        self,
        processed_data_dir='processed_data',
        output_dir='latent_representations',
        encodec_bandwidth=6.0,  # Bandwidth in kbps
        device=None,
        chunk_sizes=[5, 10]
    ):
        self.processed_data_dir = processed_data_dir
        self.output_dir = output_dir
        self.encodec_bandwidth = encodec_bandwidth
        self.chunk_sizes = chunk_sizes
        
        # Select device: CUDA if available
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        for chunk_size in chunk_sizes:
            os.makedirs(os.path.join(output_dir, f'latents_{chunk_size}s'), exist_ok=True)
            
        # Load the EnCodec model
        self._load_encodec_model()
        
    def _load_encodec_model(self):
        """Loads the EnCodec model."""
        print("Loading EnCodec model...")
        
        try:
            # Load Facebook's EnCodec model
            self.encodec_model = EncodecModel.encodec_model_24khz()
            self.encodec_model.set_target_bandwidth(self.encodec_bandwidth)
            self.encodec_model.to(self.device)
            self.encodec_model.eval()  # Set to evaluation mode
            
            print(f"EnCodec model loaded successfully (target bandwidth: {self.encodec_bandwidth} kbps)")
            
        except Exception as e:
            print(f"Error: An issue occurred while loading the EnCodec model - {str(e)}")
            raise
            
    def process_segments(self, chunk_size=5, batch_size=32, max_samples=None):
        """Generates latent representations for all audio files with the specified segment duration."""
        print(f"\nProcessing {chunk_size}s segments...")
        
        # Prepare Dataset and DataLoader
        dataset = AudioSegmentDataset(
            self.processed_data_dir, 
            chunk_size=chunk_size,
            max_samples=max_samples
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Lists to store latent representations and metadata
        all_latents = []
        all_metadata = []
        
        # Process in batches
        with torch.no_grad():  # Disable gradient computation
            for batch in tqdm(dataloader, desc=f"Encoding {chunk_size}s segments"):
                # Prepare audio waveforms
                waveforms = batch['audio'].to(self.device)
                
                # Convert each audio waveform to the expected format for EnCodec (24kHz, mono)
                waveforms_24k = []
                for waveform in waveforms:
                    wav_24k = convert_audio(
                        waveform, 
                        batch['sample_rate'][0], 
                        self.encodec_model.sample_rate, 
                        self.encodec_model.channels
                    ).to(self.device)
                    waveforms_24k.append(wav_24k)
                
                waveforms_batch = torch.stack(waveforms_24k)
                
                # Encode audio using EnCodec
                encoded_frames = self.encodec_model.encode(waveforms_batch)
                
                # Retrieve latent codes
                # The EnCodec model returns: (codes, scale)
                # codes: Dictionary format: Dict[int, Tensor] (layer -> codes)
                codes_dict = encoded_frames[0]  # This is a dictionary
                
                # For each segment, save the latent representation along with its V-A metadata
                for i in range(len(batch['file_name'])):
                    # Get the latent codes for the first layer (index 0) and convert to NumPy
                    latent = codes_dict[0][i].cpu().numpy()
                    
                    # Gather metadata
                    metadata = {
                        'song_id': int(batch['song_id'][i].item()),
                        'chunk_id': int(batch['chunk_id'][i].item()),
                        'valence': float(batch['valence'][i].item()),
                        'arousal': float(batch['arousal'][i].item()),
                        'valence_norm': float(batch['valence_norm'][i].item()),
                        'arousal_norm': float(batch['arousal_norm'][i].item()),
                        'latent_shape': latent.shape,
                        'file_name': batch['file_name'][i]
                    }
                    
                    # Save the latent representation
                    latent_file = f"{batch['song_id'][i].item()}_{batch['chunk_id'][i].item()}.npy"
                    latent_path = os.path.join(self.output_dir, f"latents_{chunk_size}s", latent_file)
                    np.save(latent_path, latent)
                    
                    # Append to the lists for statistics
                    all_latents.append(latent)
                    all_metadata.append(metadata)
        
        # Save statistics
        stats = {
            'segment_count': len(all_metadata),
            'latent_dim': all_latents[0].shape if all_latents else None,
            'chunk_size': chunk_size
        }
        
        with open(os.path.join(self.output_dir, f'latent_stats_{chunk_size}s.json'), 'w') as f:
            json.dump(stats, f, indent=2)
            
        # Save all metadata as CSV
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df.to_csv(os.path.join(self.output_dir, f'metadata_{chunk_size}s.csv'), index=False)
        
        # Analyze the latent space
        self.analyze_latent_space(all_latents, all_metadata, chunk_size)
        
        print(f"Finished processing {chunk_size}s segments! Total segments: {len(all_metadata)}")
        return all_latents, all_metadata
    
    def analyze_latent_space(self, latents, metadata, chunk_size):
        """Analyzes and visualizes the latent space."""
        print(f"\nAnalyzing latent space for {chunk_size}s segments...")
        
        # Display the shape and statistics of the latent space
        print(f"Latent code shape: {latents[0].shape}")
        
        # Flatten latent representations and reduce to 2D using t-SNE
        max_samples_tsne = min(1000, len(latents))
        
        # Compute mean and standard deviation for each latent vector (for dimensionality reduction)
        latent_stats = []
        for latent in latents[:max_samples_tsne]:
            stats = []
            for q in range(latent.shape[0]):
                q_mean = np.mean(latent[q])
                q_std = np.std(latent[q])
                stats.extend([q_mean, q_std])
            latent_stats.append(stats)
        
        latent_stats = np.array(latent_stats)
        
        # Run t-SNE dimensionality reduction
        print("Visualizing latent space with t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_stats)
        
        # Prepare metadata for visualization
        va_values = [(m['valence_norm'], m['arousal_norm']) for m in metadata[:max_samples_tsne]]
        va_array = np.array(va_values)
        
        # Overall visualization with average of valence and arousal for coloring
        plt.figure(figsize=(12, 10))
        color_values = 0.5 * (va_array[:, 0] + va_array[:, 1])
        
        scatter = plt.scatter(
            latent_2d[:, 0], 
            latent_2d[:, 1], 
            c=color_values,
            cmap='coolwarm', 
            alpha=0.7,
            s=50
        )
        
        plt.colorbar(scatter, label='Average Valence-Arousal (normalized)')
        plt.title(f'Latent Space t-SNE Visualization ({chunk_size}s segments)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'latent_tsne_{chunk_size}s.png'))
        plt.close()
        
        # Separate visualizations for valence and arousal
        plt.figure(figsize=(20, 10))
        
        # Visualization for Valence
        plt.subplot(1, 2, 1)
        scatter_v = plt.scatter(
            latent_2d[:, 0], 
            latent_2d[:, 1], 
            c=va_array[:, 0],
            cmap='RdBu_r',
            alpha=0.7,
            s=50
        )
        plt.colorbar(scatter_v, label='Valence (normalized)')
        plt.title(f'Latent Space by Valence ({chunk_size}s segments)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)

        # Visualization for Arousal
        plt.subplot(1, 2, 2)
        scatter_a = plt.scatter(
            latent_2d[:, 0], 
            latent_2d[:, 1], 
            c=va_array[:, 1],
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        plt.colorbar(scatter_a, label='Arousal (normalized)')
        plt.title(f'Latent Space by Arousal ({chunk_size}s segments)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'latent_tsne_{chunk_size}s_separate.png'))
        plt.close()
        
        # Visualization based on Valence-Arousal quadrants
        plt.figure(figsize=(20, 15))
        
        # Quadrant 1: High valence, high arousal (Cheerful)
        q1_mask = (va_array[:, 0] > 0.5) & (va_array[:, 1] > 0.5)
        plt.subplot(2, 2, 1)
        plt.scatter(latent_2d[q1_mask, 0], latent_2d[q1_mask, 1], c='#FF5733', label='High V, High A', alpha=0.7)
        plt.title('Cheerful (High Valence, High Arousal)')
        plt.grid(True, alpha=0.3)
        
        # Quadrant 2: Low valence, high arousal (Angry)
        q2_mask = (va_array[:, 0] < 0.5) & (va_array[:, 1] > 0.5)
        plt.subplot(2, 2, 2)
        plt.scatter(latent_2d[q2_mask, 0], latent_2d[q2_mask, 1], c='#C70039', label='Low V, High A', alpha=0.7)
        plt.title('Angry (Low Valence, High Arousal)')
        plt.grid(True, alpha=0.3)
        
        # Quadrant 3: Low valence, low arousal (Sad)
        q3_mask = (va_array[:, 0] < 0.5) & (va_array[:, 1] < 0.5)
        plt.subplot(2, 2, 3)
        plt.scatter(latent_2d[q3_mask, 0], latent_2d[q3_mask, 1], c='#581845', label='Low V, Low A', alpha=0.7)
        plt.title('Sad (Low Valence, Low Arousal)')
        plt.grid(True, alpha=0.3)
        
        # Quadrant 4: High valence, low arousal (Calm)
        q4_mask = (va_array[:, 0] > 0.5) & (va_array[:, 1] < 0.5)
        plt.subplot(2, 2, 4)
        plt.scatter(latent_2d[q4_mask, 0], latent_2d[q4_mask, 1], c='#FFC300', label='High V, Low A', alpha=0.7)
        plt.title('Calm (High Valence, Low Arousal)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'latent_quadrants_{chunk_size}s.png'))
        plt.close()
    
    def run(self, max_samples=None):
        """Runs the latent representation generation for all defined segment durations."""
        results = {}
        
        for chunk_size in self.chunk_sizes:
            latents, metadata = self.process_segments(
                chunk_size=chunk_size,
                max_samples=max_samples
            )
            results[chunk_size] = (latents, metadata)
            
        print("\n--- Latent Representation Generation Completed ---")
        print(f"Representations are saved in the '{self.output_dir}' directory.")
        print("Next step: VAE-based latent space manipulation")
            
        return results
        
    def test_reconstruction(self, chunk_size=5, num_examples=3):
        """Performs a reconstruction test on a few segments and saves the output."""
        print(f"\nPerforming reconstruction test on {num_examples} segments...")
        
        # Directory containing latent representations
        latent_dir = os.path.join(self.output_dir, f'latents_{chunk_size}s')
        
        # Select files for testing
        latent_files = [f for f in os.listdir(latent_dir) if f.endswith('.npy')][:num_examples]
        
        # Create an output directory for test results
        test_dir = os.path.join(self.output_dir, 'reconstruction_test')
        os.makedirs(test_dir, exist_ok=True)
        
        for latent_file in latent_files:
            # Load the latent representation
            latent = np.load(os.path.join(latent_dir, latent_file))
            
            # Extract song_id and chunk_id from the filename
            song_id, chunk_id = latent_file.replace('.npy', '').split('_')
            
            # Load the original audio file for comparison
            original_audio_path = os.path.join(
                self.processed_data_dir,
                f'chunks_{chunk_size}s',
                f'{song_id}_{chunk_id}.wav'
            )
            y_orig, sr_orig = librosa.load(original_audio_path, sr=None)
            
            # Convert the latent code to the expected format for EnCodec
            latent_tensor = torch.from_numpy(latent).to(self.device)
            
            # Prepare the codes dictionary as expected by EnCodec.decode()
            codes_dict = {0: latent_tensor.unsqueeze(0)}  # Codes for layer 0
            scale = torch.ones(1, 1).to(self.device)  # Default scaling factor
            encoded_frames = [(codes_dict, scale)]  # Format required by EnCodec
            
            # Reconstruct audio
            with torch.no_grad():
                decoded_audio = self.encodec_model.decode(encoded_frames)[0]
                
            # Move to CPU and convert to NumPy
            decoded_audio_np = decoded_audio.cpu().numpy()[0]  # mono channel
            
            # Save the reconstructed audio
            reconstructed_path = os.path.join(test_dir, f'reconstructed_{song_id}_{chunk_id}.wav')
            sf.write(reconstructed_path, decoded_audio_np, self.encodec_model.sample_rate)
            
            # Also, copy the original audio to the test directory
            original_copy_path = os.path.join(test_dir, f'original_{song_id}_{chunk_id}.wav')
            sf.write(original_copy_path, y_orig, sr_orig)
            
            # Visualize waveforms
            plt.figure(figsize=(15, 6))
            
            # Original waveform
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(y_orig, sr=sr_orig)
            plt.title(f'Original Audio: {song_id}_{chunk_id}')
            
            # Reconstructed waveform
            plt.subplot(2, 2, 1)
            librosa.display.waveshow(decoded_audio_np, sr=self.encodec_model.sample_rate)
            plt.title('Reconstructed Audio')
            
            plt.tight_layout()
            plt.savefig(os.path.join(test_dir, f'waveform_comparison_{song_id}_{chunk_id}.png'))
            plt.close()
            
        print(f"Reconstruction test completed! Results are in: {test_dir}")

    def analyze_detailed_with_librosa(self, chunk_size=5, num_examples=5):
        """Performs a detailed analysis using librosa on the original audio corresponding to latent representations.
           Computes and visualizes mel-spectrogram, MFCC, and chroma features.
        """
        import librosa.display  # Ensure librosa.display is imported
        print(f"\nPerforming detailed librosa analysis on {chunk_size}s segments for {num_examples} examples...")
        latent_dir = os.path.join(self.output_dir, f'latents_{chunk_size}s')
        audio_dir = os.path.join(self.processed_data_dir, f'chunks_{chunk_size}s')
        analysis_dir = os.path.join(self.output_dir, 'librosa_detailed_analysis')
        os.makedirs(analysis_dir, exist_ok=True)

        # Get a limited list of latent files
        latent_files = [f for f in os.listdir(latent_dir) if f.endswith('.npy')][:num_examples]

        for latent_file in latent_files:
            # Extract song_id and chunk_id from filename (assumes pattern 'songID_chunkID.npy')
            song_id, chunk_id = latent_file.replace('.npy', '').split('_')
            audio_filename = f"{song_id}_{chunk_id}.wav"
            audio_path = os.path.join(audio_dir, audio_filename)

            # Load original audio
            if not os.path.exists(audio_path):
                print(f"Audio file {audio_path} not found, skipping...")
                continue
            y, sr = librosa.load(audio_path, sr=None)

            # Compute features using librosa
            # Compute mel-spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            # Compute MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # Compute chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)

            # Plot and save the features
            plt.figure(figsize=(20, 12))

            plt.subplot(3, 1, 1)
            librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
            plt.colorbar(label='dB')
            plt.title(f'Mel-Spectrogram ({audio_filename})')

            plt.subplot(3, 1, 2)
            librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='coolwarm')
            plt.colorbar()
            plt.title('MFCC')

            plt.subplot(3, 1, 3)
            librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', cmap='coolwarm')
            plt.colorbar()
            plt.title('Chroma')

            plt.tight_layout()
            plot_file = os.path.join(analysis_dir, f'detailed_analysis_{song_id}_{chunk_id}.png')
            plt.savefig(plot_file)
            plt.close()
            print(f"Saved detailed analysis plot for {audio_filename} to {plot_file}")