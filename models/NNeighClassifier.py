import os, pickle
import numpy as np
import pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, r2_score
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq
from util.helpers import playlistToSparseMatrixEntry, getPlaylistTracks


class NNeighClassifier():
    def __init__(self, playlists, sparsePlaylists, songs, reTrain=False, name="NNClassifier.pkl"):
        self.pathName = name
        self.name = "NNC"
        self.playlistData = sparsePlaylists
        self.playlists = playlists
        self.songs = songs
        self.initModel(reTrain)


    def initModel(self, reTrain):
        """
        Initialize or load the Nearest Neighbors model.
        """
        # Specify the full path to the trained directory on your Windows system
        current_directory = os.getcwd()
        trained_dir = os.path.join(current_directory, "trained")

        # Check if the directory exists
        if not os.path.exists(trained_dir):
            # If the directory does not exist, create it
            os.makedirs(trained_dir)
            libContents = []
        else:
            # If the directory exists, list its contents
            libContents = os.listdir(trained_dir)

        # Check if the model file exists in the directory or if retraining is required
        if self.pathName not in libContents or reTrain:
            # If the model file does not exist or retraining is requested, initialize and train a new model
            self.model = NearestNeighbors(n_neighbors=60, metric="cosine")
            self.trainModel(self.playlistData)
        else:
            # If the model file exists and no retraining is needed, load the model from the file
            model_path = os.path.join(trained_dir, self.pathName)
            with open(model_path, "rb") as model_file:
                self.model = pickle.load(model_file)

    def trainModel(self, data):
        """
        """
        print(f"Training Nearest Neighbors classifier")
        self.model.fit(data)
        self.saveModel()

    def getNeighbors(self, X, k):
        """
        """
        return self.model.kneighbors(X=X, return_distance=False, n_neighbors=k)[0]

    def getPlaylistsFromNeighbors(self, neighbours, pid):
        """
        """
        neighbours = list(filter(lambda x: x not in pid.index, neighbours))
        return [self.playlists.loc[x] for x in neighbours]

    def getPredictionsFromTracks(self, tracks, numPredictions, pTracks):
        """
        """
        pTracks = set(pTracks)
        songs = defaultdict(int)
        for i, playlist in enumerate(tracks):
            if isinstance(playlist, pandas.Series):

                if playlist['Track URI'] not in pTracks:
                    songs[playlist['Track URI']] += (1 / (i + 1))
            else:
                for song in playlist.iterrows():
                    if song[1]['Track URI'] not in pTracks:
                        songs[song[1]['Track URI']] += (1 / (i + 1))



        scores = heapq.nlargest(numPredictions, songs, key=songs.get)
        return scores

    def predict(self, X, numPredictions, songs, numNeighbours=60):
        """
        x=playlist
        """
        pid, pTracks = X["Playlist ID"], X["Track URI"]
        sparseX = playlistToSparseMatrixEntry(X, self.songs)
        neighbors = self.getNeighbors(sparseX, numNeighbours)  # PlaylistIDs
        playlists = self.getPlaylistsFromNeighbors(neighbors, pid)
        # Extract all Playlist IDs from the list of Series
        playlist_ids = set([playlist['Playlist ID'] for playlist in playlists])
        tracks = []
        for id in playlist_ids:
            filtered_playlist = [playlist for playlist in playlists if playlist['Playlist ID'] == id]
            tracks.extend(getPlaylistTracks(filtered_playlist, self.songs))

        predictions = self.getPredictionsFromTracks(tracks, numPredictions, pTracks)
        return predictions

    def saveModel(self):
        """
        Saves the Nearest Neighbors model to a file.
        """
        # Define the directory where the model should be saved
        current_directory = os.getcwd()
        trained_dir = os.path.join(current_directory, "trained")

        # Ensure the directory exists
        if not os.path.exists(trained_dir):
            os.makedirs(trained_dir)

        # Define the full path for the model file
        model_path = os.path.join(trained_dir, self.pathName)

        # Save the model using a context manager to handle the file operation safely
        with open(model_path, "wb") as model_file:
            pickle.dump(self.model, model_file)
