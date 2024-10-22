import pandas as pd

# Create a DataFrame from the provided CSV-like structure
df = pd.read_csv("all_audio_features.csv")

# Modify the first column name and remove ".mp3" suffix
df.rename(columns={'filename': 'song_id'}, inplace=True)
df['song_id'] = df['song_id'].str.replace('.mp3', '').astype(int)

# Sort the DataFrame according to the 'song_id' column
df_sorted = df.sort_values(by='song_id')

# save to a new CSV file
df_sorted.to_csv("features_sorted.csv", index=False)