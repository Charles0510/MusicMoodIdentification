import os
import librosa
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Path to the directory containing audio files
audios_dir = "MEMD_audio"
audios_filename = [f for f in os.listdir(audios_dir) if f.endswith(('.mp3'))]

# Output CSV file to save all features
output_csv_file = "all_audio_features.csv"

# List to accumulate all features
all_features = []

# Function to compute statistical summaries (mean, std, min, max) of features
def summarize_features(features):
    # Compute mean, standard deviation, min, and max for each feature across the time axis (axis=1)
    return np.concatenate([
        np.mean(features, axis=1),
        np.std(features, axis=1),
        np.min(features, axis=1),
        np.max(features, axis=1)
    ])

# Function to process a single audio file and extract summarized features
def process_file(audio_filename):
    audio_path = os.path.join(audios_dir, audio_filename)

    try:
        # Load the audio file using librosa
        y, sr = librosa.load(audio_path)

        # Extract features
        # 1. MFCCs (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_summary = summarize_features(mfcc)

        # 2. Chroma (12 coefficients)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_summary = summarize_features(chroma)

        # 3. Spectral Centroid (1 coefficient)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_summary = summarize_features(spectral_centroid)

        # 4. Spectral Contrast (7 coefficients)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_summary = summarize_features(spectral_contrast)

        # 5. Zero-Crossing Rate (1 coefficient)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_summary = summarize_features(zcr)

        # 6. RMS energy (1 coefficient)
        rms = librosa.feature.rms(y=y)
        rms_summary = summarize_features(rms)

        # Concatenate all summarized features into a single feature vector
        features = np.concatenate([
            mfcc_summary,       # 52 dimensions (13 coefficients * 4 statistics)
            chroma_summary,     # 48 dimensions (12 coefficients * 4 statistics)
            spectral_centroid_summary,  # 4 dimensions (1 coefficient * 4 statistics)
            spectral_contrast_summary,  # 28 dimensions (7 coefficients * 4 statistics)
            zcr_summary,        # 4 dimensions (1 coefficient * 4 statistics)
            rms_summary         # 4 dimensions (1 coefficient * 4 statistics)
        ])

        return (audio_filename, features)

    except Exception as e:
        print(f"Error processing {audio_filename}: {e}")
        return None

# Function to process files in parallel using multi-threading
def process_files_in_parallel():
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, audios_filename))
    return results

# Start processing with multi-threading
if __name__ == "__main__":
    # Collect results
    processed_results = process_files_in_parallel()

    # Filter out failed results
    processed_results = [result for result in processed_results if result is not None]

    # Prepare a list of feature vectors and corresponding file names
    audio_files = [result[0] for result in processed_results]
    all_features = [result[1] for result in processed_results]

    # Convert the feature list into a Pandas DataFrame
    feature_columns = []
    
    # Create column names for features (e.g., 'mfcc_mean_1', 'chroma_mean_1', etc.)
    for i in [f'mfcc_mean', f'mfcc_std', f'mfcc_min', f'mfcc_max']:
        feature_columns += [f'{i}_{j+1}' for j in range(13)]
    for i in [f'chroma_mean', f'chroma_std', f'chroma_min', f'chroma_max']:
        feature_columns += [f'{i}_{j+1}' for j in range(12)]
    feature_columns += ['centroid_mean', 'centroid_std', 'centroid_min', 'centroid_max']
    for i in [f'contrast_mean', f'contrast_std', f'contrast_min', f'contrast_max']:
        feature_columns += [f'{i}_{j+1}' for j in range(7)]
    feature_columns += ['zcr_mean', 'zcr_std', 'zcr_min', 'zcr_max']
    feature_columns += ['rms_mean', 'rms_std', 'rms_min', 'rms_max']

    # Create a DataFrame with the features
    df = pd.DataFrame(all_features, columns=feature_columns)

    # Add the audio file names as the first column
    df.insert(0, 'filename', audio_files)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_file, index=False)

    print(f"All features saved in {output_csv_file}")
