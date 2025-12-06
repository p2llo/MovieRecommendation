# Movie Recommendation System (PCA + K-Means Clustering)

This project provides a local movie recommendation system built using PCA for dimensionality reduction and K-Means clustering for grouping similar movies. Only the minimal dataset and pretrained model files are required for deployment. All heavy preprocessing (ratings, tags, genome data) was performed offline and is not included in this repository.

The system allows users to filter movies by genre, minimum rating, minimum rating count (popularity), and release year range. Streamlit is used to run the interface locally.

PROJECT FILES INCLUDED:
- app.py (Streamlit application)
- movies.csv (Movie metadata: movieId, title, genres)
- movie_features.pkl (Preprocessed feature dataset used for filtering and ranking)
- kmeans_model.pkl (Trained K-Means model)
- pca_model.pkl (PCA dimensionality reduction model)
- scaler.pkl (StandardScaler used before PCA)
- requirements.txt (Python dependencies)

INSTALLATION:
Run the following command to install dependencies:
pip install -r requirements.txt

The requirements file contains:
streamlit
pandas
numpy
scikit-learn

DATA USED:
Only one dataset is needed for deployment:
movies.csv — contains movieId, title, and genres. Extracting genres and release years from titles is performed inside the app. No ratings.csv, tags.csv, genome-scores.csv, or genome-tags.csv are required for local deployment.

MODEL PREPARATION (Performed offline before deployment):
1. Cleaned movie metadata.
2. Extracted genres and release years.
3. Computed rating statistics offline and merged with movie metadata.
4. Applied scaling using StandardScaler.
5. Reduced dimensionality using PCA.
6. Clustered movies using K-Means.
7. Saved artifacts as movie_features.pkl, kmeans_model.pkl, pca_model.pkl, scaler.pkl.

RUNNING THE APP LOCALLY:
1. Clone the repository:
git clone https://github.com/p2llo/MovieRecommendation.git
cd MovieRecommendation

2. Install dependencies:
pip install -r requirements.txt

3. Run Streamlit:
streamlit run app.py

4. Access the application in your browser at:
http://localhost:8501

APP FEATURES:
The user can filter movies using:
1. Genre selection
2. Minimum average rating
3. Minimum rating count (popularity)
4. Release year range

HOW RECOMMENDATIONS WORK:
The system loads the precomputed movie features and:
- Filters by selected genre
- Applies rating threshold
- Applies popularity threshold
- Applies release year boundaries
- Sorts final results by highest mean rating and rating count
- Returns the top recommended movies

NOTES:
- This version is optimized for local deployment.
- Large preprocessing files (ratings, tags, genome data) were intentionally excluded.
- All .pkl model files and movies.csv must remain in the same folder as app.py.
- This README covers only local deployment, not cloud deployment.

