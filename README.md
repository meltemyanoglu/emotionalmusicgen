# Emotional Music Generation Project

## Project Overview
This project demonstrates a complete pipeline for generating emotional music using a Variational Autoencoder (VAE). It encompasses:
- Data Preprocessing
- Latent Representations Creation
- Model Training
- Music Generation

Each stage is organized into its respective Jupyter Notebook file.

## Environment Setup
1. Install Python 3.8+.
2. Create a new virtual environment:
   - On Windows:
     ```
     python -m venv env
     env\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     python3 -m venv env
     source env/bin/activate
     ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Execution Steps

### 1. Preprocessing (preprocessing.ipynb)
- Imports libraries and defines a DataPreparation class.
- Loads audio data and annotations.
- Analyzes the emotional distribution of the dataset.
- Processes and segments audio files into chunks (e.g., 5 and 10 seconds).
- Visualizes sample segments and analysis results.

*Run this notebook first to generate processed data for downstream tasks.*

### 2. Latent Representations Creation (create_latent_representations.ipynb)
- Uses the preprocessed data.
- Leverages an EnCodec model to produce latent representations of audio segments.
- Visualizes the latent space with sample plots.

*Execute this notebook after preprocessing to generate and inspect latent features.*

### 3. Model Training (train_model.ipynb)
- Configures training parameters (latent dimension, batch size, learning rate, etc.).
- Loads the latent representations dataset.
- Instantiates and trains the Emotional VAE model.
- Visualizes training statistics such as total loss, reconstruction loss, and KL divergence.
- Generates sample outputs from the trained model.

*Run this notebook to train the VAE and obtain model checkpoints.*

### 4. Music Generation (generate_music.ipynb)
- Loads the trained VAE model from the checkpoint.
- Sets up sample generation with varying emotional conditions.
- Converts latent samples into audio signals with defined synthesis functions.
- Saves generated audio and provides playback and visualization within the notebook.

*Execute this notebook last to generate and listen to emotional music based on the trained model.*

## Additional Notes
- Ensure all file paths within the notebooks match your local setup.
- Adjust configuration parameters (such as sample rates, output directories, and emotional conditions) as necessary.
- Comments and additional instructions in each notebook provide further guidance.
- Run the notebooks sequentially to ensure a consistent workflow and proper data flow.

## Conclusion
This repository provides a comprehensive framework for emotion-controlled music generation. From environment setup and preprocessing to training and audio synthesis, each step is detailed to facilitate a smooth implementation of the system.

Happy experimenting!
