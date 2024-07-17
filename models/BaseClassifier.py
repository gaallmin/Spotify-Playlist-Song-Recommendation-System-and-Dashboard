import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity


class BaseClassifier:
    def __init__(self, songs, playlists):
        self.songs = songs
        self.playlists = playlists
        self.sim_matrix = None
        self.prepare_data()

    def convert_embedding(self, x):
        if isinstance(x, str):
            try:
                # Strip leading/trailing brackets if they exist and any whitespace
                cleaned_x = x.strip("[] \n\t")
                # Convert string to array
                return np.fromstring(cleaned_x, sep=' ')
            except ValueError:
                print("Conversion failed for:", x)
                return np.nan
        elif isinstance(x, list):
            return np.array(x)
        return np.nan

    def prepare_data(self):
        """Prepare data by converting, cleaning, and calculating similarity matrix."""
        self.convert_embeddings()
        self.clean_data()
        self.calculate_similarity_matrix()

    def convert_embeddings(self):
        """Apply conversion to all embeddings in the dataframe."""
        self.songs['lyrics_embedding'] = self.songs['lyrics_embedding'].apply(self.convert_embedding)

    def clean_data(self):
        """Remove rows where 'lyrics_embedding' is NaN and update indices."""
        self.songs.dropna(subset=['lyrics_embedding'], inplace=True)
        print("Valid embeddings count:", len(self.songs))

    def calculate_similarity_matrix(self):
        """Calculate the cosine similarity matrix from the embeddings."""
        if not self.songs.empty:
            vectors = np.stack(self.songs['lyrics_embedding'].values)
            self.sim_matrix = cosine_similarity(vectors)
            print("Similarity matrix calculated.")
        else:
            print("No valid embeddings available to calculate similarity.")

    def index_to_uri(self, index):
        return self.songs.index[index]

    def uri_to_index(self, uri):
        return np.where(self.songs.index == uri)[0][0]


    def get_uris_in_playlist(self, playlist_id):
        playlist = self.playlists[self.playlists['Playlist ID'] == playlist_id]
        return set(playlist['Track URI'])

    def get_topk_index_sim(self, uri, k):
        index = self.uri_to_index(uri)
        sims = self.sim_matrix[index]
        top_k_indices = np.argsort(sims)[::-1][:k + 1]  # Include k+1 to skip the first identical item
        top_k_sims = sims[top_k_indices]
        return [[index, sim] for index, sim in zip(top_k_indices[1:], top_k_sims[1:])]

    def estimate_rating(self, topk, uris_in_plist):
        num = 0
        denom = 0
        for index, sim in topk:
            denom += sim
            track_uri = self.index_to_uri(index)
            if track_uri in uris_in_plist:
                num += (sim * 5)  # As rating, assuming 5 is a maximum 'like'
            else:
                num += sim
        return num / denom

    def get_recommendations(self, playlist_id, topk=10):
        uris_in_plist = self.get_uris_in_playlist(playlist_id)
        rows = []
        unique_track_uris = self.playlists['Track URI'].unique().tolist()
        for uri in unique_track_uris:
            if uri not in uris_in_plist:
                try:
                    topk_sim = self.get_topk_index_sim(uri, topk)
                    est_rating = self.estimate_rating(topk_sim, uris_in_plist)
                except Exception as e:
                    print(e)
                rows.append({'Track URI': uri, 'estimated_rating': est_rating})
        ratings_df = pd.DataFrame(rows)
        return self.provide_recs(ratings_df, topk)

    def provide_recs(self, ratings_df, k):
        ratings_df = ratings_df.sort_values(by='estimated_rating', ascending=False)
        top_k_ratings_df = ratings_df.head(k)
        rows = []
        for i, row in top_k_ratings_df.iterrows():
            uri = row['Track URI']
            rating = row['estimated_rating']
            match = self.songs[self.songs.index == uri]
            # Extract fields
            tr_name = match['Track Name'].values[0]
            art_name = match['Artist Name'].values[0]
            trk_uri = match.index.values[0]

            # Create new row for official recommendation
            new_row = {'Track URI': trk_uri, 'Track Name': tr_name, 'Artist Name': art_name,
                       'Recommendation Score': rating}

            rows.append(new_row)

        return pd.DataFrame(rows)

    def predict(self, playlist, num_predictions, songs):
        """
        Adjusted to accept playlist object and song list; generates song recommendations based on playlist ID.
        """
        # Extract playlist_id from playlist object; adjust this depending on playlist structure
        playlist_id = playlist['Playlist ID'] if isinstance(playlist, dict) else playlist['Playlist ID'].iloc[0]
        try:
            recommendations = self.get_recommendations(playlist_id, num_predictions)
            # Now, let's return recommendations in the expected format (like what predictNeighbour might expect)
            recommended_songs = songs[songs.index.isin(recommendations['Track URI'])]
            return recommended_songs.index.values
        except Exception as e:
            print(f"Error in prediction for BaseClassifier: {e}")
            return pd.DataFrame()
