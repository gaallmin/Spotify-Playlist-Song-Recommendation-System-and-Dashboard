# Import packages --------------------------------------------

import pandas as pd
import pickle

from dash import Dash, html, dcc, callback, Output, Input
from plotly.graph_objs import *
import dash_bootstrap_components as dbc
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


# Data -------------------------------------------------------------
# Access data from pkl files from data_for_dashboard.ipynb

with open('topic_track_uris.pkl', 'rb') as f:
    topic_tracks_dict = pickle.load(f)

with open('all_tracks_dict.pkl', 'rb') as f:
    all_tracks_dict = pickle.load(f)
    
with open('track_to_playlist_dict.pkl', 'rb') as f:
    track_to_playlist_dict = pickle.load(f)

playlists_songs_df = pd.read_pickle('playlists_songs_df.pkl')

with open('all_playlist_recs.pkl', 'rb') as f:
    all_playlist_recs_dict = pickle.load(f)


# List of topics (for dropdown menu)
list_of_topics = [
    "Life + Personal Struggles",
    "Relationships",
    "Self-Empowerment",
    "Nostalgic Reflection on Life",
]

# List of songs (for dropdown menu)
list_of_songs = list(all_tracks_dict.keys())

# List of playlist names (for dropdown menu)
list_of_playlists = list(all_playlist_recs_dict.keys())



# Initialize the app -----------------------------------------------
app = Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
)
app.title = "ML Final Project"


# Set up app styles ------------------------------------------------

# Note: this html establishes background color, etc.
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>{%title%}</title>
        <style>
            body, html {
                background-color: #161616; /* Set background color of the whole page for dark mode */
                color: #e6e6e6; /* Set text color for dark mode */
                font-family: Arial, sans-serif; /* Use default font */
                line-height: 1.6; /* Set line height for dark mode */
                margin: 0; /* Remove default margin */
                padding: 0; /* Remove default padding */
            }
            /* Dropdown styling */
            .Select-control {
                background-color: #333 !important; /* Set background color of dropdowns for dark mode */
                color: #e6e6e6 !important; /* Set text color of dropdowns for dark mode */
            }
            .Select-menu-outer {
                background-color: #333 !important; /* Set background color of dropdown options for dark mode */
                color: #e6e6e6 !important; /* Set text color of dropdown options for dark mode */
            }
            .Select-option {
                background-color: #333 !important; /* Set background color of dropdown options for dark mode */
                color: #e6e6e6 !important; /* Set text color of dropdown options for dark mode */
                cursor: pointer; /* Set cursor to pointer for dropdown options */
                z-index: 9999; /* Ensure dropdown options are above other elements */
            }
            .Select-option:hover {
                background-color: #555 !important; /* Set hover background color of dropdown options for dark mode */
            }
            /* Ensure text color is visible after selection */
            .Select-value, .Select-value-label {
                color: #e6e6e6 !important;
            }
            /* Your other CSS styles */
        </style>
        {%metas%}
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Set up Spotify client credentials
client_credentials_manager = SpotifyClientCredentials(client_id='da6820484d0a415b87dd46fc04a4ca66', client_secret='b99242bb9b594a7ab47159ad307573cb')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# App layout ------------------------------------------------------

app.layout = dbc.Container([
    dbc.Row([
        html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div([
                html.H4("Machine Learning Applications Final Project"),
                ],style={'display': 'inline-block'}),
            html.Div([
                html.H4("May 2024"),]
            ,style={'display': 'inline-block','textAlign': 'right', 'float': 'right'}),
        ]),
    ]),

    html.Hr(style={'borderColor': '#ffffff', 'margin': '0'}),  # horizontal line
    html.Br(),

    dbc.Row([
        html.Div([
            html.H2("Recommendation System for Spotify Playlist", style={'marginBottom': '0px'}),
            html.Hr(style={'borderColor': '#ffffff', 'margin': '0','borderWidth': '0', 'width':'10%','lineHeight': '0'}),
        ],),
        html.P(  # description for website
            "In this project, we implemented a recommendation system for playlists: Given a playlist of songs, we recommend the top songs that are similar to the playlist tracks based on semantic content (song lyrics). You can use this dashboard to explore the top songs that correlate to each of the four overall playlist themes we found in our dataset. Then, choose a song that you like, and we'll find a playlist that a user has made that contains it. You can explore what songs we recommend to add to a given playlist, including the one that contains your chosen song!",
        ),
        html.P(
            "Feel free to press play on the resulting tracks to hear them in your browser. If you like one, click on the three dots to add it to your library on Spotify!")
    ],style={'backgroundColor': '#2a2a2a', 'paddingLeft': '10px', 'paddingRight':'10px', 'paddingBottom': '10px', 'paddingTop': '10px', 'color': 'white', "width":"90%"}),



    # Explore the Top 5 Songs of Each Topic ---------------------------
    
    dbc.Row([
        html.Div(
                    className="row",
                    children=[
                        html.H2("Select a Topic:"),
                        html.Div(
                            className="div-for-dropdown",
                            children=[
                                # Dropdown for topics
                                dcc.Dropdown(
                                    options=list_of_topics,
                                    id="topics-dropdown",
                                    placeholder="Select a topic",
                                    style={"width":"300px",'text-align': 'left',}
                                )
                            ],
                        ),
                        html.Br()
                    ]
                ),

        dbc.Col([
                    html.Div(id='spotify-embeds')
                ], style={'display': 'inline-block'}),        

    ]),



    # Choose A Song ----------------------------------------------------
    
    dbc.Row([ 
        html.Div(
            id="choose_a_song",
            children=[
                html.H2("Choose a Song:"),  # header for this section
                html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        # Dropdown for topics
                                        dcc.Dropdown(
                                            list_of_songs,
                                            id="songs-dropdown",
                                            placeholder="Select a song",
                                            style={"width":"500px",'text-align': 'left',}
                                        )
                                    ],
                                ),
                            ]
                        ),
                html.Br(),
                html.Div(id="choose-song"), 
                html.Br()
            ],style={'display':'block'}),
    ]),


    # Song Recs For the Playlist -----------------------------------------

    dbc.Row([
        html.Div(id="song_recs",
            children=[
                        html.H2("Pick a Playlist:"),
                        dbc.Col([
                            html.Div(
                                        className="row",
                                        children=[
                                            html.Div(
                                                className="div-for-dropdown",
                                                children=[
                                                    # Dropdown for topics
                                                    dcc.Dropdown(
                                                        list_of_playlists,
                                                        id="playlists-dropdown-recs",
                                                        placeholder="Select a playlist",
                                                        style={"width":"300px",'text-align': 'left','margin': 'auto','margin-right': ' 50px'}
                                                    )
                                                ],
                                            ),                                            
                                        ]
                                        ,style={'display': 'flex', 'justify-content': 'left', 'align-items': 'left'}
                                    ),   
                                    html.Div([
                                        html.Br(),
                                        html.H3("And move the slider to the number of song recs you want:"),
                                        # Slider for choosing number of recs
                                        dcc.Slider(
                                            id='slider-input',
                                            min=0,
                                            max=10,
                                            step=1,
                                            value=5,)                                            
                                        ],style={'width': '33%'}),
                                    html.Div(id="song-recs")       
                                ]),
                        ]
            ,style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'left', 'justify-content': 'left'})
            ])
        ], fluid=True, 
        style={
            "font-family":"Noto Sans, sans-serif",  # change font here! list of available fonts: https://www.w3.org/Style/Examples/007/fonts.en.html
                                                    # if you want to change a font, add this to end of html block style={"font-family":"font-name"}
            #'text-align': 'left',  # text alignment/justification!
            'marginLeft': '50px', 'marginRight': '50px'  # margin setup
            })


# CALLBACKS -------------------------------------------------------------

# For "Explore the Top 5 Songs of Each Topic" section
# Update the top 5 tracks based on selection
@callback(
    Output(component_id='spotify-embeds', component_property='children'),
    Input(component_id="topics-dropdown", component_property='value')
)
def update_top5_songs(topic):
    if topic: 
        tracks = ['https://open.spotify.com/embed/track/'+ uri[14:] for uri in topic_tracks_dict[topic]]

        return html.Div([
                        html.H3("These are the top tracks associated with each topic:"),
                        html.Div([html.Iframe(src=track, width="225px",height="352px", allow="encrypted-media", style={'border':'none','paddingLeft': '5px', 'paddingRight': '5px'}) for track in tracks])
                        ])
    else:
        return


# For "Choose A Song" section
# Show a selected song once dropdown is selected
@callback(
    Output(component_id="choose-song", component_property="children"),
    Input(component_id="songs-dropdown", component_property="value")
)
def show_selected_song(song):
    if song:
        playlist_name = track_to_playlist_dict.get(song, None)
        if playlist_name:
            playlist_df = playlists_songs_df[playlists_songs_df['Playlist Name'] == playlist_name]

            # Create table
            column_widths = {'Track Name': '50%', 'Artist Name': '25%', 'Album Name': '25%'}
            headers = [html.Th(col, style={'width':column_widths.get(col, 'auto')}) for col in ['Track Name', 'Artist Name', 'Album Name']]
            header_row = html.Tr(headers)

            # Create table rows
            rows = [html.Tr([html.Td(playlist_df.iloc[i][col]) for col in ['Track Name', 'Artist Name', 'Album Name']]) for i in range(len(playlist_df))]

            # Combine headers and rows into a table
            table = html.Div(
                    [html.H3("We found your chosen song in this playlist: "+ playlist_name),
                     html.P("You can scroll through the playlist here."),
                        html.Div(
                                html.Table([header_row]+ [*rows], 
                                    style={'border': 'none', 'textAlign':'left',})
                        ,style={'height': '300px', 'overflowY': 'scroll', 'width':'60%'})
                     ])

            return table
        else:
            return html.Div("No playlist found for this song.")
    else:
        return


# For "Song Recs For the Playlist"
# Show recommended songs for the playlist once playlist is selected
@callback(
    Output(component_id="song-recs", component_property='children'),
    Input(component_id="playlists-dropdown-recs", component_property='value'),
    Input(component_id="slider-input", component_property="value")
)
def show_playlist_recs(playlist, value):
    if playlist in list_of_playlists:
        tracks = ['https://open.spotify.com/embed/track/'+ uri[14:] for uri in all_playlist_recs_dict[playlist]]
        return html.Div([
                        html.H3("If you like this playlist, we think you'll like these " + str(value) + " songs:"),
                        html.Div([html.Iframe(src=tracks[i-1], width="225px",height="352px", allow="encrypted-media", style={'border':'none','paddingLeft': '0px', 'paddingRight': '10px'}) for i in range(value)])
                        ])
    return


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
