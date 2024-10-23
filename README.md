# MusicMoodIdentification
Detect music mood

Dataset: DEAM dataset https://cvml.unige.ch/databases/DEAM/

## Dataset and Annotations

The annotations of DEAM dataset contains "valence_mean", "valence_std", "arousal_mean", "arousal_std". Please refer to `Dataset_manual.pdf` for more details.The values of these annotations are continuous rather than discrete, which means that the annotations are not directly suitable for classification tasks. There are two options: the first is to train a regression model to predict the continuous values of the annotations, and the second is to discretize the annotations into several classes. Here I choose the first option. (can also try the second option if have time)

I use the song-level notations in the file `annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv` and `annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv`. I take "valence_mean" and "arousal_mean" as the target values of the regression model because they reflect the overall mood of the music.

For model training and validation, I split the dataset into training set (80%) and test set (20%). 

## Features

In order to apply machine learning methods to detect music mood (no matter it is a classification task or a regression task), we need to extract features from the musics and have to ensure the features' dimensions are consistent across all the musics. However, the musics in the DEAM dataset are of different lengths, which makes some features (e.g. MFCC) have different dimensions across the musics. To solve this problem, I further extract the statistics of the features (mean, std, min, max) and use them as the features of the musics. The features provided by Mingcheng in `main_attempt3.py` are:

- 13 MFCCs
- 12 Chroma features
- 1 Spectral centroid
- 7 Spectral contrast
- 1 Zero-crossing rate
- 1 RMS energy

The features are extracted using the `librosa` library in Python. For each of them, the mean, std, min, max are further extracted as the final features. In total, there are (13 + 12 + 1 + 7 + 1 + 1) x 4 = 140 features for each music. The code for the final feature extraction is in `extract-features.py`.

## Regression model

I have tried two regression models: Gradient Boosting Regressor and XGBoost. I use grid search to find the best hyperparameters for the models and the two targets. A five-fold cross-validation is used to evaluate the models and the $R^2$ metric is used in the grid searching. In conclusion, XGBoost has a slightly better performance with higher $R^2$ score and lower mean squared error on the test dataset. However, both models appear to be overfitting and only have a $R^2$ score from 0.4 to 0.6 on the testing data. Maybe we can try more/less features or other models to improve the performance.

Please check more details and results in `regression.ipynb`.

## Classification model

Though have not tried yet, but SVM, Gradient Boosting, XGBoost, Random Forest, and Neural Network are available options for classification models. These are future work.