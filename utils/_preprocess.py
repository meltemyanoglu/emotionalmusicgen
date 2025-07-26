import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import soundfile as sf
import json
from tqdm import tqdm


class DataPreparation:
    def __init__(
        self, 
        annotations_path='data\\annotations',
        audio_path='data\\audio',
        output_path='processed_data',
        sample_rate=16000,
        chunk_sizes=[5, 10]  # 5 and 10-second segments
    ):
        self.annotations_path = annotations_path
        self.audio_path = audio_path
        self.output_path = output_path
        self.sample_rate = sample_rate  # Standard sample rate
        self.chunk_sizes = chunk_sizes  # Segment lengths in seconds
        
        # Create output folders
        os.makedirs(output_path, exist_ok=True)
        for chunk_size in chunk_sizes:
            os.makedirs(os.path.join(output_path, f'chunks_{chunk_size}s'), exist_ok=True)
            os.makedirs(os.path.join(output_path, f'metadata_{chunk_size}s'), exist_ok=True)
        
        # Load the dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Loads and prepares the DEAM dataset"""
        print("Loading dataset...")
        
        # Load static and dynamic annotations
        self.static_annotations = pd.read_csv(
            os.path.join(self.annotations_path, 
                         'annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv')
        )
        
        self.dynamic_annotations_arousal = pd.read_csv(
            os.path.join(self.annotations_path, 
                        'annotations averaged per song/dynamic (per second annotations)/arousal.csv')
        )
        
        self.dynamic_annotations_valence = pd.read_csv(
            os.path.join(self.annotations_path, 
                        'annotations averaged per song/dynamic (per second annotations)/valence.csv')
        )
        
        # Clean unnecessary spaces in column names
        self.static_annotations.columns = self.static_annotations.columns.str.strip()
        self.dynamic_annotations_arousal.columns = self.dynamic_annotations_arousal.columns.str.strip()
        self.dynamic_annotations_valence.columns = self.dynamic_annotations_valence.columns.str.strip()
        
        print(f"Total {len(self.static_annotations)} songs found.")
    
    def analyze_dataset(self):
        """Performs basic analysis and visualization of the dataset"""
        print("\nDataset Analysis:")
        print("-" * 40)
        
        # Show basic statistics
        print("Valence-Arousal Distribution:")
        print(self.static_annotations[['valence_mean', 'arousal_mean']].describe())
        
        # Valence-Arousal scatter plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x='valence_mean', 
            y='arousal_mean', 
            data=self.static_annotations,
            alpha=0.7,
            s=60
        )
        plt.title('Emotional Distribution (Valence-Arousal)')
        plt.xlabel('Valence (Positive/Negative Emotion)')
        plt.ylabel('Arousal (Energy Level)')
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.grid(True)
        
        # Add guide lines in the middle of the plot
        plt.axhline(y=5, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
        
        # Label the quadrants
        plt.text(7.5, 7.5, "Positive / High Energy\n(happy, excited)", ha='center')
        plt.text(2.5, 7.5, "Negative / High Energy\n(angry, tense)", ha='center')
        plt.text(7.5, 2.5, "Positive / Low Energy\n(calm, peaceful)", ha='center')
        plt.text(2.5, 2.5, "Negative / Low Energy\n(sad, depressed)", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'valence_arousal_distribution.png'))
        plt.show()

    def _chunk_song(self, y, sr, chunk_size_seconds):
        """Splits an audio file into segments of the specified length"""
        chunk_size_samples = chunk_size_seconds * sr
        num_chunks = int(len(y) / chunk_size_samples)
        chunks = []
        
        for i in range(num_chunks):
            start = int(i * chunk_size_samples)
            end = int((i + 1) * chunk_size_samples)
            chunk = y[start:end]
            chunks.append(chunk)
            
        return chunks
    
    def _normalize_and_save_va_data(self):
        """Normalizes and saves Valence-Arousal values"""
        # For each segment size
        for chunk_size in self.chunk_sizes:
            # Convert to DataFrame
            va_df = pd.DataFrame(self.valence_arousal_data[chunk_size])
            
            if not va_df.empty:
                # Min-max normalization (0-1 range)
                va_df['valence_normalized'] = (va_df['valence'] - va_df['valence'].min()) / (va_df['valence'].max() - va_df['valence'].min())
                va_df['arousal_normalized'] = (va_df['arousal'] - va_df['arousal'].min()) / (va_df['arousal'].max() - va_df['arousal'].min())
                
                # Save metadata for each segment
                for _, row in va_df.iterrows():
                    metadata = {
                        'song_id': int(row['song_id']),
                        'chunk_id': int(row['chunk_id']),
                        'valence': float(row['valence']),
                        'arousal': float(row['arousal']),
                        'valence_normalized': float(row['valence_normalized']),
                        'arousal_normalized': float(row['arousal_normalized'])
                    }
                    
                    json_path = os.path.join(
                        self.output_path, 
                        f'metadata_{chunk_size}s', 
                        f'{int(row["song_id"])}_{int(row["chunk_id"])}.json'
                    )
                    
                    with open(json_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                
                # Visualize summary statistics
                plt.figure(figsize=(10, 8))
                sns.scatterplot(
                    x='valence_normalized', 
                    y='arousal_normalized', 
                    data=va_df,
                    alpha=0.7,
                    s=40
                )
                plt.title(f'Normalized Emotional Distribution ({chunk_size}s segments)')
                plt.xlabel('Valence (Normalized)')
                plt.ylabel('Arousal (Normalized)')
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.grid(True)
                
                # Add guide lines in the middle of the plot
                plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
                
                # Label the quadrants
                plt.text(0.75, 0.75, "Positive / High Energy\n(happy, excited)", ha='center')
                plt.text(0.25, 0.75, "Negative / High Energy\n(angry, tense)", ha='center')
                plt.text(0.75, 0.25, "Positive / Low Energy\n(calm, peaceful)", ha='center')
                plt.text(0.25, 0.25, "Negative / Low Energy\n(sad, depressed)", ha='center')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path, f'valence_arousal_normalized_{chunk_size}s.png'))
                plt.show()
                
                # Save summary statistics
                stats_file = os.path.join(self.output_path, f'stats_{chunk_size}s.json')
                stats = {
                    'segment_count': len(va_df),
                    'unique_songs': va_df['song_id'].nunique(),
                    'valence_stats': {
                        'min': float(va_df['valence'].min()),
                        'max': float(va_df['valence'].max()),
                        'mean': float(va_df['valence'].mean()),
                        'std': float(va_df['valence'].std())
                    },
                    'arousal_stats': {
                        'min': float(va_df['arousal'].min()),
                        'max': float(va_df['arousal'].max()),
                        'mean': float(va_df['arousal'].mean()),
                        'std': float(va_df['arousal'].std())
                    }
                }
                
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                print(f"\nProcessing completed for {chunk_size}s segments:")
                print(f"- Total number of segments: {len(va_df)}")
                print(f"- Number of unique songs: {va_df['song_id'].nunique()}")


    def process_audio_files(self):
        """Processes audio files and splits them into segments"""
        print("\nProcessing audio files...")
        
        # Dictionaries to store results for all segment sizes
        self.all_chunks = {chunk_size: {} for chunk_size in self.chunk_sizes}
        self.valence_arousal_data = {chunk_size: [] for chunk_size in self.chunk_sizes}
        
        # Get the list of song IDs to process
        song_ids = self.static_annotations['song_id'].tolist()
        
        # Process each song
        for song_id in tqdm(song_ids, desc="Processing songs"):
            # Path to the audio file
            audio_file = os.path.join(self.audio_path, f"{song_id}.mp3")
            
            # Skip if the file does not exist
            if not os.path.exists(audio_file):
                continue
            
            try:
                # Load the audio file and convert to mono
                y, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
                
                # Get dynamic valence and arousal values
                valence_dynamic = self.dynamic_annotations_valence[self.dynamic_annotations_valence['song_id'] == song_id]
                arousal_dynamic = self.dynamic_annotations_arousal[self.dynamic_annotations_arousal['song_id'] == song_id]
                
                # If no dynamic annotations are available, use static annotations
                if valence_dynamic.empty or arousal_dynamic.empty:
                    # Get static annotations
                    static_data = self.static_annotations[self.static_annotations['song_id'] == song_id]
                    if static_data.empty:
                        continue
                    
                    valence = float(static_data['valence_mean'].values[0])
                    arousal = float(static_data['arousal_mean'].values[0])
                    
                    # Process for each segment size
                    for chunk_size in self.chunk_sizes:
                        chunks = self._chunk_song(y, sr, chunk_size)
                        self.all_chunks[chunk_size][song_id] = chunks
                        
                        # Use the same V-A values for each segment
                        for i, chunk in enumerate(chunks):
                            self.valence_arousal_data[chunk_size].append({
                                'song_id': song_id,
                                'chunk_id': i,
                                'valence': valence,
                                'arousal': arousal
                            })
                            
                            # Save the audio segment
                            chunk_path = os.path.join(
                                self.output_path, 
                                f'chunks_{chunk_size}s', 
                                f'{song_id}_{i}.wav'
                            )
                            sf.write(chunk_path, chunk, self.sample_rate)
                else:
                    # Process using dynamic annotations
                    time_columns = [col for col in valence_dynamic.columns if col.startswith('sample_')]
                    time_values = np.array([int(col.split('_')[1][:-2]) / 1000 for col in time_columns])
                    
                    valence_values = valence_dynamic[time_columns].values[0]
                    arousal_values = arousal_dynamic[time_columns].values[0]
                    
                    # Process for each segment size
                    for chunk_size in self.chunk_sizes:
                        chunks = self._chunk_song(y, sr, chunk_size)
                        self.all_chunks[chunk_size][song_id] = chunks
                        
                        # Calculate average V-A values for each segment
                        for i, chunk in enumerate(chunks):
                            start_time = i * chunk_size
                            end_time = (i + 1) * chunk_size
                            
                            # Find the appropriate time indices for this segment
                            time_indices = np.where((time_values >= start_time) & (time_values < end_time))[0]
                            
                            if len(time_indices) > 0:
                                segment_valence = np.mean(valence_values[time_indices])
                                segment_arousal = np.mean(arousal_values[time_indices])
                            else:
                                # If no time indices are found, use static values
                                static_data = self.static_annotations[self.static_annotations['song_id'] == song_id]
                                segment_valence = float(static_data['valence_mean'].values[0])
                                segment_arousal = float(static_data['arousal_mean'].values[0])
                            
                            self.valence_arousal_data[chunk_size].append({
                                'song_id': song_id,
                                'chunk_id': i,
                                'valence': segment_valence,
                                'arousal': segment_arousal
                            })
                            
                            # Save the audio segment
                            chunk_path = os.path.join(
                                self.output_path, 
                                f'chunks_{chunk_size}s', 
                                f'{song_id}_{i}.wav'
                            )
                            sf.write(chunk_path, chunk, self.sample_rate)
                
            except Exception as e:
                print(f"Error: An issue occurred while processing {song_id} - {str(e)}")
        
        # Normalize and save V-A data
        self._normalize_and_save_va_data()
    
    def visualize_chunk_examples(self, num_examples=3):
        """Displays examples of segments from each quadrant"""
        for chunk_size in self.chunk_sizes:
            # Read JSON files
            metadata_path = os.path.join(self.output_path, f'metadata_{chunk_size}s')
            json_files = [f for f in os.listdir(metadata_path) if f.endswith('.json')]
            
            if not json_files:
                continue
            
            # Read all metadata
            all_metadata = []
            for json_file in json_files[:500]:  # Limit for performance
                with open(os.path.join(metadata_path, json_file), 'r') as f:
                    metadata = json.load(f)
                    all_metadata.append(metadata)
            
            # Convert to DataFrame
            metadata_df = pd.DataFrame(all_metadata)
            
            # Select examples from each quadrant
            # Quadrant 1: High valence, high arousal (happy)
            q1 = metadata_df[(metadata_df['valence_normalized'] > 0.75) & (metadata_df['arousal_normalized'] > 0.75)]
            
            # Quadrant 2: Low valence, high arousal (angry)
            q2 = metadata_df[(metadata_df['valence_normalized'] < 0.25) & (metadata_df['arousal_normalized'] > 0.75)]
            
            # Quadrant 3: Low valence, low arousal (sad)
            q3 = metadata_df[(metadata_df['valence_normalized'] < 0.25) & (metadata_df['arousal_normalized'] < 0.25)]
            
            # Quadrant 4: High valence, low arousal (calm)
            q4 = metadata_df[(metadata_df['valence_normalized'] > 0.75) & (metadata_df['arousal_normalized'] < 0.25)]
            
            # Display examples from each quadrant
            quarters = [q1, q2, q3, q4]
            quarter_names = ["Happy", "Angry", "Sad", "Calm"]
            
            plt.figure(figsize=(15, 20))
            
            for i, (quarter, name) in enumerate(zip(quarters, quarter_names)):
                if len(quarter) > 0:
                    # Select num_examples examples from each quadrant
                    samples = quarter.sample(min(num_examples, len(quarter)))
                    
                    for j, (_, sample) in enumerate(samples.iterrows()):
                        song_id = int(sample['song_id'])
                        chunk_id = int(sample['chunk_id'])
                        
                        # Load the audio file
                        audio_path = os.path.join(self.output_path, f'chunks_{chunk_size}s', f'{song_id}_{chunk_id}.wav')
                        
                        if os.path.exists(audio_path):
                            # Load the audio file
                            y, sr = librosa.load(audio_path, sr=None)
                            
                            # Plot waveform
                            plt.subplot(4, num_examples, i*num_examples + j + 1)
                            librosa.display.waveshow(y, sr=sr)
                            plt.title(f"{name}: V={sample['valence_normalized']:.2f}, A={sample['arousal_normalized']:.2f}")
                            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, f'example_segments_{chunk_size}s.png'))
            plt.show()