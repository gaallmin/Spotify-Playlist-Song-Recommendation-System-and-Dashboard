import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import dok_matrix
import os


def parseTrackURI(uri):
    return uri.split(":")[2]


def processPlaylistForClustering(playlists, tracks):
    """
    Create sparse matrix mapping playlists to track
    lists that are consumable by most clustering algos
    """

    # List of all track IDs in db
    trackIDs = list(tracks["Track URI"])

    # Map track id to matrix index
    IDtoIDX = {k: v for k, v in zip(trackIDs, range(len(trackIDs)))}

    playlistIDs = list(playlists["Playlist ID"])

    print("Create sparse matrix mapping playlists to tracks")
    playlistSongSparse = dok_matrix((len(playlistIDs), len(trackIDs)), dtype=np.float32)

    for i in tqdm(range(len(playlistIDs))):
        # Get playlist and track ids from DF
        playlistID = playlistIDs[i]
        trackID = playlists.loc[playlistID]["Track URI"]
        playlistIDX = playlistID

        # Get matrix index for track id
        trackIDX = [IDtoIDX.get(i) for i in trackID]

        # Remove None values from trackIDX
        trackIDX = [idx for idx in trackIDX if idx is not None]

        # Set index to 1 if playlist has song
        playlistSongSparse[playlistIDX, trackIDX] = 1

    return playlistSongSparse.tocsr(), IDtoIDX


def createDFs(path, idx, num_files):
    """
    Creates playlist and track DataFrames from
    json files
    """
    final_data = pd.read_pickle(path)
    sliced_data = final_data.iloc[idx:idx + num_files]

    # Splitting data into playlists and tracks DataFrames
    playlist_df = sliced_data[['Playlist Name', 'Playlist ID', 'Track URI']].copy()
    tracks_df = sliced_data[['Track Name','Track URI', 'Artist Name', 'Playlist ID','lyrics_embedding']].copy()

    # Optionally, you can reset index for both DataFrames
    playlist_df.reset_index(drop=True, inplace=True)
    tracks_df.reset_index(drop=True, inplace=True)

    # os.makedirs("data/final", exist_ok=True)
    playlist_df.set_index("Playlist ID")

    # Split id from spotifyURI for brevity
    tracks_df = tracks_df.sample(frac=1).reset_index(drop=True)
    playlist_df = playlist_df.sample(frac=1).reset_index(drop=True)
    tracks_df["Track URI"] = tracks_df.apply(lambda row: parseTrackURI(row["Track URI"]), axis=1)
    playlist_df["Track URI"] = playlist_df.apply(lambda row: parseTrackURI(row["Track URI"]), axis=1)

    playlistClusteredDF, IDtoIDXMap = processPlaylistForClustering(playlists=playlist_df,
                                                                   tracks=tracks_df)

    # Add sparseID for easy coercision to sparse matrix for training data
    tracks_df["sparse_id"] = 0
    tracks_df["sparse_id"] = tracks_df.apply(lambda row: IDtoIDXMap[row["Track URI"]], axis=1)
    tracks_df = tracks_df.set_index("Track URI")

    # Check for duplicate indices
    print(playlist_df.index.duplicated().sum())
    print(tracks_df.index.duplicated().sum())

    # Get the count of each index
    print(playlist_df.index.value_counts())

    # Check if all playlist Track URIs are in tracks DataFrame
    print(playlist_df['Track URI'].isin(tracks_df.index).all())

    # Alternatively, find which are not present
    missing_uris = playlist_df.loc[~playlist_df['Track URI'].isin(tracks_df.index), 'Track URI']
    print(missing_uris)

    # Check index after resetting
    playlist_df.reset_index(drop=True, inplace=True)
    print(playlist_df.index)

    # Write DFs to CSVs
    print(f"Pickling {len(playlist_df)} playlists")
    print(playlist_df.head(20))
    print(tracks_df.head(20))
    # Get the current working directory
    current_directory = os.path.join(os.getcwd())

    # Construct the full file path
    playlist_path = os.path.join(current_directory, "data", "playlists.pkl")
    playlist_df.to_pickle(playlist_path)
  

    print(f"Pickling {len(tracks_df)} tracks")
    print(tracks_df.head(20))
    tracks_path = os.path.join(current_directory, "data", "tracks.pkl")
    tracks_df.to_pickle(tracks_path)
    print(f"Pickling clustered playlist")
    try:
        playlistSparse_path = os.path.join(current_directory, "data", "playlistSparse.pkl")

        with open(playlistSparse_path, "wb") as f:
            pickle.dump(playlistClusteredDF, f)
        print(f"File {playlistSparse_path} saved successfully.")
    except Exception as e:
        print(f"Failed to save file at {playlistSparse_path}: {e}")


