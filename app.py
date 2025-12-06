import streamlit as st
import pandas as pd
import pickle
import re

# -----------------------------------------------------
# LOAD MODELS AND DATA
# -----------------------------------------------------
@st.cache_resource
def load_models():
    movie_features = pickle.load(open("movie_features.pkl", "rb"))
    movies = pd.read_csv("movies.csv")
    return movie_features, movies

movie_features, movies = load_models()

# -----------------------------------------------------
# EXTRACT YEAR FROM TITLE
# -----------------------------------------------------
def extract_year(title):
    year_match = re.search(r"\((\d{4})\)", title)
    return int(year_match.group(1)) if year_match else None

movies["year"] = movies["title"].apply(extract_year)

# -----------------------------------------------------
# GET UNIQUE GENRES
# -----------------------------------------------------
genre_set = set()
for g_list in movies["genres"]:
    for g in g_list.split("|"):
        genre_set.add(g)

genres = sorted(list(genre_set))

# -----------------------------------------------------
# MULTI-FEATURE RECOMMENDER
# -----------------------------------------------------
def recommend_movies(selected_genre, min_rating, min_rating_count, year_range, n=10):

    df = movies.merge(movie_features, on="movieId", how="inner")

    # Detect correct genre column
    if "genres" in df.columns:
        genre_col = "genres"
    elif "genres_x" in df.columns:
        genre_col = "genres_x"
    elif "genres_y" in df.columns:
        genre_col = "genres_y"
    else:
        raise KeyError("No genre column found in dataset.")

    # 1. Genre filter
    df = df[df[genre_col].str.contains(selected_genre, case=False, na=False)]

    # 2. Rating filter
    df = df[df["mean_rating"] >= min_rating]

    # 3. Popularity filter
    df = df[df["rating_count"] >= min_rating_count]

    # 4. Year filter
    df = df[df["year"].between(year_range[0], year_range[1], inclusive="both")]

    # Sort results
    df = df.sort_values(by=["mean_rating", "rating_count"], ascending=False)

    # Return dynamic genre column, not fixed "genres"
    return df.head(n)[["title_x", genre_col, "mean_rating", "rating_count", "year"]]


# -----------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------
st.title("Multi-Feature Movie Recommendation System (PCA + KMeans)")

st.subheader("Choose Recommendation Filters")

# Feature 1: Genre
selected_genre = st.selectbox("Select Genre:", genres)

# Feature 2: Minimum Rating
min_rating = st.slider("Minimum Rating:", 0.0, 5.0, 3.5, 0.1)

# Feature 3: Minimum Popularity (rating count)
min_rating_count = st.slider("Minimum Rating Count:", 0, 5000, 100)

# Feature 4: Release Year Filter
year_min = int(movies["year"].min()) if movies["year"].notna().any() else 1900
year_max = int(movies["year"].max()) if movies["year"].notna().any() else 2023

year_range = st.slider("Release Year Range:", year_min, year_max, (1990, 2023))

if st.button("Get Recommendations"):
    results = recommend_movies(selected_genre, min_rating, min_rating_count, year_range)
    
    st.write("Top Recommendations:")
    st.table(results)
