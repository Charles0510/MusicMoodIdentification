# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:55:07 2024

@author: Mingcheng Kou

# TODO: Description of the project

Dataset: DEAM dataset


"""



import pandas as pd
import os
import librosa


"""

Data Preprocessing
    - Read CSV files
    - Read audio files

"""
# %% Data load

# read CSV files
AnV_dir_2to2000 = "annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
AnV_dir_2000to2058 = "annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv"

AnV1 = pd.read_csv(AnV_dir_2to2000)
AnV2 = pd.read_csv(AnV_dir_2000to2058)

AnV = pd.concat([AnV1, AnV2], ignore_index=True) # combine the data into one dataframe variable


# %% Extract features
audios_dir = "MEMD_audio"
audios_filename = [f for f in os.listdir(audios_dir) if f.endswith(('.mp3'))]
mfcc_list = []
chroma_list = []
spectral_centroid_list = []
spectral_contrast_list = []
zcr_list = []
rms_list = []


for audio_filename in audios_filename:
    audio_path = os.path.join(audios_dir, audio_filename)

    # Load the audio file
    y, sr = librosa.load(audio_path)
    
    # extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_list.append(mfcc)
    
    # extract Chorma
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_list.append(chroma)
    
    # extract spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_list.append(spectral_centroid)
    
    # extract spectral contrast 
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_list.append(spectral_contrast)
    
    # extract zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_list.append(zcr)
    
    # extract RMS energy
    rms = librosa.feature.rms(y=y)
    rms_list.append(rms)

        