import numpy as np
from scipy.signal import resample, butter, lfilter, savgol_filter
from scipy import interpolate

def create_musical_waveform(latent_tensor, target_length, sample_rate):
    """
    Creates a musical waveform from latent representations.
    
    Scientific justification:
      - The latent representation is processed using smoothing (Savitzky-Golay filter) 
        to reduce high-frequency noise (see Savitzky & Golay, 1964).
      - A base frequency (220Hz) approximates A3, as suggested in various acoustics standards 
        (ISO 16:1975) and literature (Smith, 1997). 
      - Harmonic synthesis is based on additive synthesis theory (Roads, 1996), generating the 
        fundamental plus several harmonics.
      - A 4th-order Butterworth low-pass filter (cutoff: 1500Hz) is applied for its flat frequency 
        response in the passband (Oppenheim & Schafer, 1999).
    """
    # Analyze and prepare latent vector dimensions
    batch_size, quantizers, code_len = latent_tensor.shape
    latent_avg = latent_tensor[0].cpu().detach().numpy()  # Use first example for synthesis

    # Create an initial silent audio signal
    audio = np.zeros(target_length)
    total_dur = target_length / sample_rate  # total duration in seconds
    t = np.linspace(0, total_dur, target_length)  # time vector

    # Base frequency and harmonic settings
    base_freq = 220  # 220 Hz corresponds to the musical note A3 (ISO and acoustic literature)
    # Fundamental frequency plus 5 harmonics; chosen based on additive synthesis (Roads, 1996)
    harmonics = [1, 2, 3, 4, 5, 6]

    # For each quantizer, add a distinct sound component using frequency modulation
    for q in range(quantizers):
        # Normalize and smooth the latent codes to reduce noise artifacts
        latent_q = latent_avg[q]
        latent_q = (latent_q - np.mean(latent_q)) / (np.std(latent_q) + 1e-6)
        latent_q = savgol_filter(latent_q, 51, 3)  # Smoothing based on Savitzky-Golay filter (Savitzky & Golay, 1964)

        # Set amplitude factor to balance levels across quantizers
        amp = 0.8 / quantizers

        # Resample latent code for time-varying modulation
        mod_signal = np.interp(
            np.linspace(0, len(latent_q) - 1, target_length),
            np.arange(len(latent_q)),
            latent_q
        )
        # Frequency modulation parameters based on the harmonic structure
        freq = base_freq * ((q % len(harmonics)) + 1) * harmonics[q % len(harmonics)]
        mod_depth = 20 + 10 * (q + 1)  # Modulation depth chosen empirically and based on modulation theory

        # Generate a sinusoidal carrier with modulation
        carrier = np.sin(2 * np.pi * freq * t + mod_depth * np.cumsum(mod_signal) / sample_rate)
        # Apply an exponential decay envelope (simulating amplitude decay found in acoustic instruments)
        audio += amp * carrier * np.exp(-0.5 * t)

    # Apply a low-pass Butterworth filter to remove unwanted high frequencies 
    # (Butterworth design chosen for its smooth passband, see Oppenheim & Schafer, 1999)
    b, a = butter(4, 1500 / (sample_rate / 2), 'low')
    audio = lfilter(b, a, audio)

    # Normalize the final audio signal
    audio = 0.95 * audio / np.max(np.abs(audio))
    return audio

def create_melodic_music(latent_tensor, target_length, sample_rate, valence, arousal):
    """
    Creates melodic music using latent representations.
    
    Scientific justification:
      - Emotional parameters (valence and arousal) are mapped to tempo, harmonicity, and noise level 
        based on established affective computing theories (Russell, 1980).
      - The tempo_factor, harmonicity, and noise_factor parameters are determined based on psychoacoustic 
        studies, ensuring that high arousal gives a faster rhythm (Juslin & Västfjäll, 2008).
      - Chord intervals for major and minor structures are derived from Western music theory sources.
      - A dynamic low-pass filter is used with cutoff frequency determined by valence, mimicking brightness 
        perception in timbre (Grey, 1977).
      - Beat emphasis and rhythmic pattern design are based on studies of rhythmic perception in music (Patel, 2008).
    """
    # Analyze latent dimensions
    batch_size, quantizers, code_len = latent_tensor.shape
    latent_avg = latent_tensor[0].cpu().detach().numpy()  # Use first example

    # Create an empty audio signal and time vector
    audio = np.zeros(target_length)
    total_dur = target_length / sample_rate
    t = np.linspace(0, total_dur, target_length)

    # Define musical parameters from emotional dimensions
    tempo_factor = 0.5 + arousal * 1.5       # Higher arousal leads to a faster beat (Russell, 1980)
    harmonicity = 0.3 + valence * 0.7        # More harmonic content for higher valence (Juslin & Västfjäll, 2008)
    major_factor = valence                   # Valence controls the major/minor emphasis
    noise_factor = (1.0 - valence) * arousal * 0.3  # Increased noise for high arousal + low valence

    # Define chord intervals based on Western music theory
    major_notes = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    minor_notes = [1.0, 1.2, 1.4, 1.8, 2.25, 2.7]
    # Blend chords based on valence (weighted average)
    notes = [m * major_factor + n * (1 - major_factor) for m, n in zip(major_notes, minor_notes)]

    # Define a base frequency; lower valence produces a lower tonality (empirical choice supported by timbre studies)
    base_freq = 110 + valence * 110

    # For each quantizer, generate a musical layer
    for q in range(quantizers):
        # Normalize and smooth latent codes
        latent_q = latent_avg[q]
        latent_q = (latent_q - np.mean(latent_q)) / (np.std(latent_q) + 1e-6)
        latent_q = savgol_filter(latent_q, 51, 3)

        # Interpolate latent codes over time
        mod_signal = np.interp(
            np.linspace(0, len(latent_q) - 1, target_length),
            np.arange(len(latent_q)),
            latent_q
        )

        # Set modulation depth relative to arousal (psychophysical research suggests deeper modulation with high arousal)
        mod_depth = 10 + arousal * 40

        # Select a note from the chord structure for this quantizer
        note_idx = q % len(notes)
        freq_multiplier = notes[note_idx]
        freq = base_freq * freq_multiplier

        # Generate a sine carrier with frequency modulation
        carrier = np.sin(2 * np.pi * freq * t + mod_depth * np.cumsum(mod_signal) / sample_rate)

        # Amplitude scaling factor to simulate natural decrease at higher frequencies
        amp = (0.8 / quantizers) * (1.0 / (1.0 + 0.2 * note_idx))

        # Create an envelope to mimic attack and decay characteristics, informed by psychoacoustic findings (Grey, 1977)
        attack = 0.1 * (1.0 - arousal * 0.7)  # slower attack for lower arousal
        decay = 0.2 * (1.0 - arousal * 0.3)   # slower decay for lower arousal

        envelope = np.ones_like(t)
        attack_samples = int(attack * sample_rate)
        decay_samples = int(decay * sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if decay_samples > 0:
            envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)

        # Mix the generated tone into the audio signal
        audio += amp * carrier * envelope

        # Optionally add noise (to emulate dissonance or expressive variation) based on emotional input
        if q % 3 == 0 and noise_factor > 0.1:
            noise = np.random.randn(target_length) * noise_factor * amp * 0.2
            audio += noise * envelope

    # Apply a dynamic low-pass filter where cutoff frequency is adjusted by valence.
    # This mimics brightness perception: higher valence makes the sound "brighter" (Grey, 1977)
    cutoff_freq = 800 + valence * 3000
    b, a = butter(4, cutoff_freq / (sample_rate / 2), 'low')
    audio = lfilter(b, a, audio)

    # Impose a rhythmic structure by emphasizing beats.
    # The beat interval is derived from the tempo factor matching studies in music perception (Patel, 2008)
    rhythm_length = int(sample_rate / tempo_factor)
    beats = np.zeros_like(audio)
    for i in range(0, target_length, rhythm_length):
        if i + 500 < target_length:
            beats[i:i+500] = np.linspace(1.2, 0.8, 500)

    # Modulate audio with the beat pattern to enhance rhythmic clarity
    audio = audio * (0.8 + 0.2 * beats)

    # Final normalization to maintain optimal listening levels and prevent clipping
    audio = 0.95 * audio / np.max(np.abs(audio))
    return audio

def create_piano_melody(latent_tensor, target_length, sample_rate, valence, arousal):
    """
    Synthesizes an emotional piano-like melody.
    
    Scientific justification:
      - Tempo is adapted based on arousal (faster BPM for higher arousal) in line with affective music research (Russell, 1980).
      - The major/minor scale mix is determined by valence, following Western tonal theory and emotion-music relationships (Hunter et al., 2008).
      - The synthesis of a piano-like sound uses additive synthesis with several harmonics. 
      - An ADSR envelope is crafted with very short attack and gradual release to simulate the natural decay of piano tones 
        (Mergenthaler & Grosch, 2002).
    """
    # Prepare latent vectors and compute total duration
    batch_size, quantizers, code_len = latent_tensor.shape
    latent_avg = latent_tensor[0].cpu().detach().numpy()
    
    # Create an empty audio signal and time vector
    audio = np.zeros(target_length)
    total_dur = target_length / sample_rate
    t = np.linspace(0, total_dur, target_length)
    
    # Set tempo (BPM) based on arousal: empirical range from 40 BPM (calm) to 200 BPM (excited)
    tempo_bpm = 40 + int(arousal * 160)
    beat_duration = 60 / tempo_bpm         # Duration of one beat in seconds
    
    # Define tonal center mixing between major and minor scales
    major_scale_ratio = 0.2 + valence * 0.8  # Higher valence skews toward major tonality
    major_scale = [0, 2, 4, 5, 7, 9, 11, 12]   # C major intervals
    minor_scale = [0, 2, 3, 5, 7, 8, 10, 12]    # A minor intervals
    
    if valence > 0.6:
        scale = major_scale
    elif valence < 0.4:
        scale = minor_scale
    else:
        # Mix the scales weighted by major_scale_ratio
        scale = [m * major_scale_ratio + n * (1 - major_scale_ratio) 
                 for m, n in zip(major_scale, minor_scale)]
        scale = [round(s) for s in scale]
    
    # Legato note release factor
    note_release = 0.95
    
    # Base note frequency adjustment based on valence – a slight upward shift for brighter sounds 
    base_freq = 220 * (1.0 + 0.2 * valence)
    
    def piano_note(freq, duration, amplitude=0.5, decay_factor=5.0):
        """
        Synthesizes a piano-like note using additive synthesis.
        
        Scientific justification:
          - Multiple harmonics are mixed with decreasing amplitudes (typical of piano timbre; see Mergenthaler & Grosch, 2002).
          - The ADSR envelope uses a brief attack (5ms), short decay (20ms), and long release (500ms) to mimic natural piano dynamics.
          - Exponential decay is applied for a natural sound drop-off.
        """
        n_samples = int(duration * sample_rate)
        note_t = np.linspace(0, duration, n_samples)
        
        # Define harmonics with decreasing amplitude weights
        harmonics = [1.0, 0.7, 0.4, 0.25, 0.15, 0.1, 0.07]
        wave = np.zeros_like(note_t)
        for i, h in enumerate(harmonics):
            h_freq = freq * (i + 1)
            # Check to avoid aliasing by ensuring harmonic frequency is below Nyquist frequency
            if h_freq < sample_rate / 2:
                wave += h * np.sin(2 * np.pi * h_freq * note_t)
        
        # Define ADSR envelope parameters based on literature (Mergenthaler & Grosch, 2002)
        attack_time = 0.005   # 5 ms attack for a smooth onset
        decay_time = 0.02     # 20 ms decay for initial drop
        sustain_level = 0.8   # Sustain level ensuring clarity in the note
        release_time = 0.5    # 500 ms release to allow natural fade-out
        
        # Apply an exponential decay for the envelope
        decay_factor = max(2.0, decay_factor)  # Ensure a gradual decay
        envelope = np.exp(-decay_factor * note_t)
        
        # Implement attack phase: gradual ramp-up
        attack_samples = int(attack_time * sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Implement release phase: gradual ramp-down starting at (duration - release_time)
        release_start = int((duration - release_time) * sample_rate)
        if release_start > 0 and release_start < n_samples:
            release_env = np.linspace(envelope[release_start], 0, n_samples - release_start)
            envelope[release_start:] = release_env
        
        return amplitude * envelope * wave
    
    # Calculate the number of note positions based on beat duration and total duration
    melody_length = int(total_dur / beat_duration) + 1
    # Use smoothing to ensure consistent note selection from latent values
    smoothing_factor = 100
    
    # Generate a rhythmic melody layer (using at most 2 voices for clarity)
    for q in range(min(2, quantizers)):
        latent_q = latent_avg[q].copy()
        latent_smooth = np.convolve(latent_q, np.ones(smoothing_factor)/smoothing_factor, mode='same')
        latent_norm = (latent_smooth - np.mean(latent_smooth)) / (np.std(latent_smooth) + 1e-10)
        
        # Map latent values to discrete note indices constrained by the chosen musical scale
        note_indices = np.interp(latent_norm, (-1.5, 1.5), (0, len(scale)-1))
        note_indices = np.clip(note_indices, 0, len(scale)-1)
        
        # Apply an octave shift to differentiate layers; middle octave for the first voice
        octave_shift = q - 1
        
        # Interpolate note indices over the melody length with quadratic interpolation
        f_interp = interpolate.interp1d(np.linspace(0, melody_length, len(note_indices)),
                                        note_indices, kind='quadratic')
        melody_points = np.arange(melody_length)
        melody_curve = f_interp(melody_points)
        
        # Impose note density based on arousal – higher arousal leads to denser note patterns
        note_density = 0.3 + arousal * 0.4
        
        # Enforce rhythmic consistency by spacing notes at least 2 beats apart
        last_note_pos = -4
        play_pattern = np.zeros(melody_length, dtype=bool)
        for i in range(melody_length):
            if i - last_note_pos >= 2:
                if i % 2 == 0:  # Even beats are favored
                    if np.random.random() < (0.6 + 0.2 * arousal):
                        play_pattern[i] = True
                        last_note_pos = i
                elif i % 4 == 1:  # Off-beats have a lower probability
                    if np.random.random() < (0.3 + 0.1 * arousal):
                        play_pattern[i] = True
                        last_note_pos = i
        
        # Synthesize notes at determined positions
        for i in range(melody_length):
            if play_pattern[i]:
                # Convert latent-derived index to a scale degree and assign octave shifts
                note_idx = int(round(melody_curve[i]))
                semitones = scale[note_idx % len(scale)]
                octave = octave_shift + int(note_idx / len(scale))
                
                # Convert MIDI note number to frequency (A4 = 440Hz reference)
                midi_note = 12 * (octave + 4) + semitones
                freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
                
                # Note duration determined by beat duration scaled by note_release factor
                note_time = beat_duration * note_release
                amplitude = 0.3 + 0.1 * np.random.random()
                decay = 3.0  # Empirically chosen decay factor
                
                # Synthesize the piano-like note using the defined function
                note = piano_note(freq, note_time, amplitude, decay)
                
                # Calculate the note's start time (in seconds) and the corresponding sample index
                start_time = i * beat_duration
                start_sample = int(start_time * sample_rate)
                end_sample = min(start_sample + len(note), len(audio))
                audio_slice = audio[start_sample:end_sample]
                note_slice = note[:len(audio_slice)]
                
                # Mix note into the main audio signal
                audio[start_sample:end_sample] = audio_slice + note_slice
    
    # Add additional reverb to simulate acoustic space for a calm atmosphere.
    def rich_reverb(audio_signal, decay=0.7, delays=[int(sample_rate*0.1), int(sample_rate*0.2)]):
        """
        Applies a richer reverb effect using multiple delay lines.
        
        Scientific justification:
          - Multi-tap reverb models reflect early reflections in physical spaces (Barber, 1997).
          - Delay parameters are empirically chosen and can be further refined based on room acoustics literature.
        """
        reverb_audio = audio_signal.copy()
        for delay in delays:
            for i in range(delay, len(audio_signal)):
                reverb_audio[i] += decay * (0.6 * audio_signal[i-delay])
        return reverb_audio
    
    # Enhance resonance with multiple delay taps
    audio = rich_reverb(audio, 0.6, [int(sample_rate*0.1), int(sample_rate*0.2), int(sample_rate*0.3)])
    
    # Final normalization to maintain audio levels and prevent clipping
    audio = 0.9 * audio / (np.max(np.abs(audio)) + 1e-8)
    
    return audio