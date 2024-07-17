# Spotify-Playlist-Song-Recommendation-System-and-Dashboard

## Authors
 Shivani Manivasagan, Min Jegal, Alba Arribas Cervan, Ana Real Terradez

## Summary
This project implements and evaluates several recommendation systems for Spotify playlists. We developed a content-based recommendation system using song lyrics and a user-based collaborative filtering system based on playlist overlap. The dataset created includes ~10,000 tracks from 100 playlists, preprocessed and vectorized using Bag-of-Words (BoW) and GloVe embeddings.
![](https://github.com/gaallmin/Spotify-Playlist-Song-Recommendation-System-and-Dashboard/blob/main/figs/dash_overview1.PNG)
![](https://github.com/gaallmin/Spotify-Playlist-Song-Recommendation-System-and-Dashboard/blob/main/figs/dash_overview2.PNG)

## Table of Contents
- [Dataset Creation](#dataset-creation)
- [Text Preprocessing and Vectorization](#text-preprocessing-and-vectorization)
- [Recommendation Systems](#recommendation-systems)
- [Dashboard](#dashboard)
- [Acknowledgements](#acknowledgements)

## Dataset Creation
1. **Criteria:**
   - Overlapping tracks across playlists.
   - Lyrics for each track.
   - At least 10,000 tracks.

2. **Method:**
   - Used the Million Playlist Dataset and the LRCLIB API for lyrics.
   - Filtered playlists to meet criteria, resulting in 79 playlists with a mean of 123 tracks each.

3. **Exploratory Analysis:**
   - Final dataset: 9720 tracks, 79 playlists.
   - Issues: Low overlap between playlists and high token variance in lyrics.

## Text Preprocessing and Vectorization
1. **Text Preprocessing Pipeline:**
   - Tokenization, homogenization, cleaning (using NLTK).

2. **Corpus and Dictionary Filtering:**
   - Removed infrequent and extremely frequent tokens.
   - Analyzed unique token occurrences and implemented N-gram detection.

3. **Vectorization Techniques:**
   - BoW for LDA topic modeling, finding 4 optimal topics.
   - GloVe for word embeddings, used in clustering and similarity calculations.

## Recommendation Systems
1. **Content-Based Recommender System:**
   - Used cosine similarity of GloVe embeddings.
   - Low accuracy due to data characteristics and lack of overlap.

2. **User-Based Collaborative Filtering System:**
   - Sparse matrix creation, playlist similarity metrics using cosine distance.
   - K-Nearest Neighbors for recommendations.
   - Multiple optimizations for recommendation accuracy, including weighted frequency and exclusion of test playlists.

3. **Performance Comparison:**
   - Evaluated using accuracy of obscured track hits.
   - Collaborative filtering outperformed content-based recommendations.

## Dashboard
- Developed using Python Dash.
- Features: 
  - Explore top songs by themes.
  - Playlist-based song recommendations.
  - Interactive parameter adjustments for recommendations.

## Acknowledgements
- **Professors:**
  - Carlos Sevilla Salcedo
  - Jerónimo Arenas García
  - Vanessa Gómez Verdejo

- **External Resources:**
  - Various academic papers and GitHub repositories on music recommendation systems and machine learning techniques.

## Links
- [Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
- [LRCLIB API Documentation](https://lrclib.net/docs)

---
