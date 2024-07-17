import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_histogram(ax, title, xlabel, ylabel, data):
    try:
        ax.hist(data, density=True)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    except Exception as e:
        print(f"Error plotting histogram: {e}")

def plot_bar_chart(ax, title, xlabel, ylabel, x_positions, heights, labels):
    try:
        ax.bar(x_positions, heights)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation="vertical", fontsize=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    except Exception as e:
        print(f"Error plotting bar chart: {e}")

def displayPopularArtists(df, limit=100):
    try:
        artist_counts = {}
        for playlist in df['Track URI']:
            for song in playlist:
                artist_name = song["Artist Name"]
                if artist_name in artist_counts:
                    artist_counts[artist_name] += 1
                else: 
                    artist_counts[artist_name] = 1
        
        sorted_artists = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        artists, counts = zip(*sorted_artists)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_bar_chart(ax, "Number of Playlist Appearances by Top Artists",
                       "Artist", "Number of Appearances", np.arange(limit), counts, artists)
        plt.savefig("figs/popular_artists.png")
    except Exception as e:
        print(f"Error in display_popular_artists: {e}")

def displayMostCommonKeyWord(df):
    try:
        playlist_names = df['Playlist Name'].tolist()
        keywords = [word for name in playlist_names for word in name.split() if word]
        keyword_counts = {word: keywords.count(word) for word in set(keywords)}
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        words, counts = zip(*sorted_keywords)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        plot_bar_chart(ax, "Most Common Words in Playlist Titles",
                       "Keyword", "Frequency", np.arange(len(words)), counts, words)
        plt.savefig("figs/keyword_frequency.png")
    except Exception as e:
        print(f"Error in display_most_common_keywords: {e}")

def displayPlaylistLengthDistribution(df):
    try:
        playlist_lengths = [len(playlist) for playlist in df['Track URI']]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_histogram(ax, "Distribution of Number of Tracks per Playlist",
                       "Number of Tracks", "Distribution", playlist_lengths)
        plt.savefig("figs/playlist_length_distribution.png")
    except Exception as e:
        print(f"Error in display_playlist_length_distribution: {e}")
