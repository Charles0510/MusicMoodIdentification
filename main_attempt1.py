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

for audio_filename in audios_filename:
    audio_path = os.path.join(audios_dir, audio_filename) # get the path to single audio files
    audio_mp3.append(pydub.AudioSegment.from_file(audio_path, format="mp3")) # read queries mp3 files to a list


# %% Sample audio

sample_rate = 44100 
samples_list = []
chunk_size = 1024 * 1024 # process 1Mb chunks at a time

for a in audio_mp3:
    a.set_channels(1) # the audio files have 2 channels, now we merge into 1
    
    samples = np.array([], dtype=np.float32) # initialize samples
    
    num_samples = len(a.get_array_of_samples())
    
    # normalize samples, process in small chunks 
    for start in range(0, num_samples, chunk_size):
        chunk = np.array(a.get_array_of_samples()[start:start + chunk_size])
        
        # the datatype of audio samples is int16
        chunk = chunk.astype(np.float32) / np.iinfo(np.int16).max
        
        samples = np.append(samples, chunk)
        
    # add new sample to list
    samples_list.append(samples)
    
    del samples
    gc.collect()
    
# %% Calclulate STFT

stft_list = []

#for s in samples_list:
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    