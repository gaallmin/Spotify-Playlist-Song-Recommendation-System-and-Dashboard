import os
import random

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from models.BaseClassifier import BaseClassifier
from models.NNeighClassifier import NNeighClassifier
from util import dataIn
from util.helpers import getTrackandArtist, obscurePlaylist


class SpotifyExplorer:
    """
    Args:
        numFiles (int): CLI variable that determines how many MPD files to read
        retrainNNC (bool): determines whether to retrain NNC or read from file

    Attributes:
        NNC (NNeighClassifier): NNeighbor Classifier used for predictions
        baseClassifier (BaseClassifier): Baseline classifier for comparison
        playlists (DataFrame): contains all playlists read into memory
        songs (DataFrame): all songs read into memory
        playlistSparse (scipy.CSR matrix) playlists formatted for predictions
    """

    def __init__(self, numFiles, retrainNNC=True):
        self.readData(numFiles)
        self.buildClassifiers(retrainNNC)

    def buildClassifiers(self, retrainNNC):
        """
        Init classifiers and set initial classifier as main
        """
        self.NNC = self.buildNNC(retrainNNC)
        self.baseClassifier = self.buildBaseClassifier()
        self.classifier = self.NNC

    def buildNNC(self, shouldRetrain):
        """
        Init NNC classifier
        """
        self.NNC = NNeighClassifier(
            sparsePlaylists=self.playlistSparse,
            songs=self.songs,
            playlists=self.playlists,
            reTrain=shouldRetrain)
        return self.NNC

    def buildBaseClassifier(self):
        """
        Init base classifier
        """
        self.baseClassifier = BaseClassifier(
            songs=self.songs,
            playlists=self.playlists)
        return self.baseClassifier

    def setClassifier(self, classifier="NNC"):
        """
        Select classifier to set as main classifier
        """
        if classifier == "NNC":
            self.classifier = self.NNC
        elif classifier == "Base":
            self.classifier = self.baseClassifier

    def readData(self, numFilesToProcess):
        """
        Read song and playlist data
        Either read from MPD data or pickled dataframe
        """
        # Get the current working directory
        current_directory = os.getcwd()

        # Construct the full file path
        file_path = os.path.join(current_directory, "playlist.pkl")
        # don't have to write every time
        if numFilesToProcess > 0:
            path = os.path.join(current_directory, "data", "playlist_with_embeddings_dataset.pkl")
            dataIn.createDFs(path, idx=0, num_files=numFilesToProcess)

        # Read data
        print("Reading data")
        # Ensure the directory exists, you may not need to create it manually if it's guaranteed to exist
        os.makedirs(os.path.join(current_directory, "/data"), exist_ok=True)
        self.playlists = pd.read_pickle(os.path.join(current_directory, "data", "playlists.pkl"))
        self.songs = pd.read_pickle(os.path.join(current_directory, "data", "tracks.pkl"))
        self.songs = self.songs[self.songs != '1fnuyUQC4OLHLjapBWKeKv']
        self.playlistSparse = pd.read_pickle(os.path.join(current_directory, "data", "playlistSparse.pkl"))
        print(f"Working with {len(self.playlists)} playlists " + f"and {len(self.songs)} songs")

    def getRandomPlaylist(self):
        playlist_id = self.playlists.sample().get("Playlist ID").iloc[0]
        # Filter the DataFrame for rows where 'Playlist ID' matches the specific ID
        return self.playlists[self.playlists['Playlist ID'] == playlist_id]

        # return self.playlists.iloc[random.randint(0, len(self.playlists) - 1)]

    def predictNeighbour(self, playlist, numPredictions, songs):
        """
        Use currently selected predictor to predict neighborings songs
        """
        return self.classifier.predict(playlist, numPredictions, songs)

    def obscurePlaylist(self, playlist, obscurity):
        """
        Obscure a portion of a playlist's songs for testing
        """
        k = len(playlist['Track URI']) * obscurity // 100
        indices = random.sample(range(len(playlist['Track URI'])), k)
        obscured = [playlist['Track URI'][i] for i in indices]
        tracks = [i for i in playlist['Track URI'] + obscured if i not in playlist['Track URI'] or i not in obscured]
        return tracks, obscured

    def evalAccuracy(self, numPlaylists, percentToObscure=0.15):
        """
        Obscures a percentage of songs
        Iterates and sees how many reccomendations match the missing songs
        """
        print()
        print(f"Selecting {numPlaylists} playlists to test and obscuring {int(percentToObscure * 100)}% of songs")

        def getAcc(pToObscure):
            playlist = self.getRandomPlaylist()

            keptTracks, obscured = obscurePlaylist(playlist, pToObscure)
            playlistSub = playlist.copy()
            obscured = set(obscured)
            # playlistSub['Track URI'] = keptTracks
            playlistSub = playlistSub[playlistSub['Track URI'].isin(keptTracks)]
            predictions = self.predictNeighbour(playlistSub,
                                                k,
                                                self.songs)

            overlap = set(predictions) & set(obscured)

            return len(overlap) / len(obscured) if obscured else 0

        accuracies = [getAcc(percentToObscure) for _ in tqdm(range(numPlaylists))]
        avgAcc = round(sum(accuracies) / len(accuracies), 4) * 100
        print(f"We predicted {avgAcc}% of obscured songs")
        return avgAcc

    def displayRandomPrediction(self):
        playlist = self.getRandomPlaylist()
        while len(playlist["Track URI"]) < 10:
            playlist = self.getRandomPlaylist()

        predictions = self.predictNeighbour(playlist,
                                            50,
                                            self.songs)

        playlistName = playlist["Playlist Name"]
        playlist = [getTrackandArtist(trackURI, self.songs) for trackURI in playlist["Track URI"]]
        predictions = [getTrackandArtist(trackURI, self.songs) for trackURI in predictions["Track URI"]]
        return {
            "Playlist Name": playlistName,
            "Playlist": playlist,
            "Predictions": predictions
        }

    def createRandomPredictionsDF(self, numInstances):
        print(f"Generating {numInstances} data points")
        data = [self.displayRandomPrediction() for _ in tqdm(range(numInstances))]
        df = pd.DataFrame(data)
        df.to_csv("predictionData.csv")
        print("Information correctly saved into a csv file")


if __name__ == "__main__":
    # Parse command line arguments
    # numToParse = int(input("Enter the number of files to parse: "))
    """
    Builds explorer
    numFiles: Number of files to load (each with 1000 playlists)
    parse:    Boolean to load in data
    """
    numToParse = 1000
    # Init class
    spotify_explorer = SpotifyExplorer(numToParse)

    k_values = range(1, 101)
    accuracies_NNC = []
    accuracies_Base = []

    for k in k_values:
        print("NNC")
        spotify_explorer.setClassifier("NNC")
        accuracy_NNC = spotify_explorer.evalAccuracy(100, percentToObscure=0.25)
        accuracies_NNC.append(accuracy_NNC)

        print("Base")
        spotify_explorer.setClassifier("Base")
        accuracy_Base = spotify_explorer.evalAccuracy(30, percentToObscure=0.25)
        accuracies_Base.append(accuracy_Base)

    # Plotting
    plt.plot(k_values, accuracies_NNC, label='NNC')
    plt.plot(k_values, accuracies_Base, label='Base')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for different values of k')
    plt.legend()
    plt.show()

    # # Generate prediction CSV
    # spotify_explorer.createRandomPredictionsDF(100)
