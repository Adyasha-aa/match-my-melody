import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load dataset
df = pd.read_csv("Hindi_songs.csv")
df.fillna(0, inplace=True)

# Convert duration (mm:ss) to seconds
def convert_to_seconds(time_str):
    try:
        minutes, seconds = map(int, str(time_str).split(':'))
        return minutes * 60 + seconds
    except:
        return 0

df['duration'] = df['duration'].apply(convert_to_seconds)

# Select features
features = ['duration', 'danceability', 'acousticness', 'energy', 'liveness', 'Valence']
df_features = df[features]

# Standardize features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)


# Fit k-NN model
knn = NearestNeighbors(n_neighbors=11, metric='cosine')
knn.fit(df_scaled)

# Streamlit config
st.set_page_config(page_title="🎵 MatchMyMelody", page_icon="🎶", layout="centered")
st.markdown("<h1 style='text-align: center; color: #6a0dad;'>🎵 MatchMyMelody</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Discover songs with a similar vibe 🎶</p>", unsafe_allow_html=True)

# Input from user
song_name = st.text_input("🔍 Enter a song name:")

# Helper functions
def find_index_by_song_name(song_name):
    matches = df[df['song_name'].str.lower() == song_name.lower()]
    return matches.index[0] if not matches.empty else None

def recommend_songs(index, n_recommendations=5):
    distances, indices = knn.kneighbors([df_scaled[index]])
    recs = []
    for i in range(1, n_recommendations + 1):
        idx = indices[0][i]
        song = df.iloc[idx]
        recs.append({
            "🎵 Song": song['song_name'],
            "🎤 Artist": song.get('singer', 'Unknown'),
            "🌐 Language": song.get('language', 'N/A'),
            "📅 Released": song.get('released_date', 'N/A'),
            "⚡ Energy": round(song.get('energy', 0), 2)
        })
    return pd.DataFrame(recs)

# Recommendation trigger
if st.button("✨ Recommend"):
    index = find_index_by_song_name(song_name)
    if index is not None:
        st.success(f"Similar songs to '{song_name.title()}':")
        st.dataframe(recommend_songs(index))
    else:
        st.error("Song not found. Try another name.")
