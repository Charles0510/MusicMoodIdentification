# Development Log
## Dataset
We give up using  MTG-Jamendo dataset. reason: 
- too large, audio files over 100 Gb
- have multiple tags of mood/theme, SVM can’t handle such multiple tags classification     
So we  switch to DEAM dataset: 
DEAM: 
- index from 2 to 2058, but actually 1802 songs in total 
- 45 seconds segment
- sample frequency 44100HZ
- contains arousal and valence information

## Dependencies
- librosa: choose librosa but not pydub, because we have familiar with basic audio processing procedure, and librosa can provide some integrated tools to extract features
- pandas: process csv files 
- os: get file names 
- #TODO: other dependencies

## Data preprocessing
### 1st attemption: 
- using pydub
- apply STFT to each audio file
- process in chunks of 1Mb size
encountered memory error: the variable that store audio file, STFT has captured very large memory space(nearly 30 GB when process till nearly 1000 songs, and exceed the memory of my computer)

### 2st attemption: 
- use pydub
- apply STFT to each audio file 
- process in chunks of 1Mb size
- define function of preprocessing and run in batches(500 files per batch)
- export the STFT files to pickles files and store in disk rather than in memory
Memory error has been remedied, but each pickles file will capture like 7GB, in total will be in nearly 30GB, that’s hard for collaborating.

### 3st attemption: 
- use librosa
- directly extract the features from audio files, don’t need to store STFT results in memory
No memory error, no need to export pickle files

## Feature extraction
The features extracted:
- MFCCs
- Chroma
- Spectral centroid
- Spectral contrast
- Zero-crossing rate

