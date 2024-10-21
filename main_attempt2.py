# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:55:07 2024

@author: Mingcheng Kou

# TODO: Description of the project

Dataset: DEAM dataset


"""



import pandas as pd
import os
import pydub
from pydub.utils import which
from scipy import signal
import numpy as np
import gc
import pickle


"""

Data Preprocessing
    - Read CSV files
    - Read audio files

"""
# %% Data load
pydub.AudioSegment.ffmpeg = which("ffmpeg")
pydub.AudioSegment.ffprobe = which("ffprobe")

# read CSV files
AnV_dir_2to2000 = "annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
AnV_dir_2000to2058 = "annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv"

AnV1 = pd.read_csv(AnV_dir_2to2000)
AnV2 = pd.read_csv(AnV_dir_2000to2058)

AnV = pd.concat([AnV1, AnV2], ignore_index=True) # combine the data into one dataframe variable


# read audio(mp3) files
audios_dir = "MEMD_audio"
audios_filename = [f for f in os.listdir(audios_dir) if f.endswith(('.mp3'))]
audio_mp3 = []

chunk_size = 1024 * 1024
sample_rate = 44100 

# Function to process a batch of files
def process_batch(start_index, end_index, save_path):
    stft_list = []  # Initialize list to store STFT results

    for i, audio_filename in enumerate(audios_filename[start_index:end_index]):
        audio_path = os.path.join(audios_dir, audio_filename)

        # Load the audio file and convert it to mono
        audio = pydub.AudioSegment.from_file(audio_path, format="mp3")
        audio = audio.set_channels(1)

        chunks = []  # Initialize list to store chunks

        num_samples = len(audio.get_array_of_samples())

        # Process the audio in smaller chunks
        for start in range(0, num_samples, chunk_size):
            chunk = np.array(audio.get_array_of_samples()[start:start + chunk_size])
            chunk = chunk.astype(np.float32) / np.iinfo(np.int16).max  # Normalize the chunk
            chunks.append(chunk)  # Store the chunk in the list

        # Concatenate all chunks into a single array
        samples = np.concatenate(chunks)

        # Calculate STFT
        f, t, Zxx = signal.stft(samples, fs=sample_rate, nperseg=1024)

        # Store the STFT result
        stft_list.append(Zxx)

        # Free up memory after each file
        del audio, samples, chunk, chunks
        gc.collect()

    # Save the results to disk
    with open(save_path, 'wb') as f:
        pickle.dump(stft_list, f)

    # Clear memory after saving
    del stft_list
    gc.collect()
        
# Process the first batch 
process_batch(0, 500, 'stft_batch1.pkl')

# Process the second batch 
process_batch(500, 1000, 'stft_batch2.pkl')

# Process the thrid batch 
process_batch(1000, 1500, 'stft_batch3.pkl')

# Process the forth batch 
process_batch(1500, len(audios_filename), 'stft_batch4.pkl')





        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    