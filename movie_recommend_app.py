import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load KNN model and movie dataset
loaded_model = pickle.load(open('movie_recommender.sav', 'rb'))
df = pd.read_csv('movies_recommendation_data.csv')  # Make sure this includes genre and title columns

# Sidebar/Input
st.title("🎬 Movie Recommender")

# 1. Movie title input
movie_title = st.text_input("Enter a movie title:")

# 2. IMDB rating slider
rating = st.slider("IMDB Rating (0–10):", 0.0, 10.0, 5.0, step=0.1)

# 3. Genre selection (assuming you have a known list of genres)
all_genres = ['Biography', 'Drama', 'Thriller', 'Comedy', 'Crime', 'Mystery', 'History']  # Update this to match your dataset
selected_genres = st.multiselect("Select genre(s):", all_genres)

# Button to generate recommendations
if st.button("Get Recommendations"):
    if not movie_title or not selected_genres:
        st.warning("Please fill out all fields before submitting.")
    else:
        # --- Sample preprocessing based on your model's expected input format ---

        # Dummy encoding genres (adjust based on how your model was trained)
        genre_vector = [1 if genre in selected_genres else 0 for genre in all_genres]

        # Combine inputs into one feature vector
        single_sample = np.array([rating] + genre_vector).reshape(1, -1)

        # Get nearest neighbors
        distances, indices = loaded_model.kneighbors(single_sample, n_neighbors=5)
        recommended_movies = df.iloc[indices[0], 1]  # Assuming title is in column index 1

        # Display recommendations
        st.subheader(f"🎥 Recommended Movies Similar to: *{movie_title}*")
        for title, dist in zip(recommended_movies, distances[0]):
            st.write(f"**{title}** (Distance: {dist:.3f})")

