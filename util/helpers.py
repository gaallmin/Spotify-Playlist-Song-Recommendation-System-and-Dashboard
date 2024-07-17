import random
from scipy.sparse import dok_matrix


def playlistToSparseMatrixEntry(playlist, songs):
    """
    Converts a playlist with a list of songs into a sparse matrix with just one row.
    """
    if 'Track URI' not in playlist:
        print("Track_id column missing in playlist.")
        return None

    # Create a sparse matrix with dimensions 1 x (max index of sparse_id + 1)
    max_sparse_id = songs['sparse_id'].max()
    playlistMtrx = dok_matrix((1, max_sparse_id + 1), dtype=int)

    for track_id in playlist['Track URI']:
        if track_id in songs.index:
            sparse_id = songs.at[track_id, 'sparse_id']
            playlistMtrx[0, sparse_id] = 1
        else:
            print(f"Track ID {track_id} not found in songs DataFrame.")

    return playlistMtrx.tocsr()


def getPlaylistTracks(playlist, songs):
    tracks = []
    songs['Track URI'] = songs.index

    for x in playlist:
        try:
            track = songs.loc[x['Track URI']]
            tracks.append(track)
        except KeyError:
            print(f"Track ID {x} not found in the songs DataFrame.")
    return tracks


def getTrackandArtist(trackURI, songs):
    try:
        song = songs.loc[str(trackURI)]
        return (song["Track Name"], song["Artist Name"])
    except KeyError:
        print(f"Track URI {trackURI} not found in songs DataFrame")
        return None


def obscurePlaylist(playlist, percentToObscure):
    """
    Obscure a portion of a playlist's songs for testing
    """
    total_tracks = len(playlist['Track URI'])
    k = int(total_tracks * percentToObscure)  # Number of tracks to obscure

    indices = random.sample(playlist.index.tolist(), k)  # Randomly pick indices to obscure


    # Now create a list of tracks that are not obscured
    mask = playlist.index.isin(indices)

    # Create a Series of obscured tracks
    obscured = playlist.loc[mask, 'Track URI']

    # Create a Series of tracks that are not obscured using the inverse of the mask
    tracks = playlist.loc[~mask, 'Track URI']
    return tracks, obscured
