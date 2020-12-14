# -*- coding: utf-8 -*-
"""
Downloads, stores, and processes NHL play-by-play reports into individual data frames for each game.

Created on Fri Nov 27 13:16:44 2020

@author: Nathan Wodarz
"""

from pathlib import Path
import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
import pickle
import re
from collections import Counter
import numpy as np
import logging
logging.basicConfig(filename='logs.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

#%% Constants
# Event location data was first added to the game feeds in the 2010-2011 season. Consequently, that will be the oldest season
# used in this project.
# The last season used will be the 2018-19 season. This season is chosen because it's the most-recent non-COVID season and because
# shot location information in the 2019-20 season was incorrect. For details, see https://www.ontheforecheck.com/2019/10/15/20915205/nhl-shot-location-play-by-play-data-has-changed-pbp-shotmaps-analytics-expected-goals-model

# Stem for NHL Stats API
API_ROOT_URL = 'https://statsapi.web.nhl.com'
# For local storage.
DATA_FOLDER = 'data/'
GAME_FRAME_FOLDER = DATA_FOLDER + 'games/'
RAW_FOLDER = DATA_FOLDER + 'raw/'
RAW_LIVE_FEED_FOLDER = RAW_FOLDER + 'feeds/'
RAW_HTML_REPORT_FOLDER = RAW_FOLDER + 'html/'
# List of seasons to use.
SEASON_LIST = ['20102011', '20112012', '20122013', '20132014', '20142015', '20152016', '20162017', 
               '20172018', '20182019', '20192020']
# Event codes in the live feed are somewhat verbose. This dictionary provides a base to translate
# the codes to match those given in the play-by-play HTML report.
EVENT_TRANSLATION = {
    'Game Scheduled': None, 
    'Period Ready': None, 
    'Period Start': 'PSTR', 
    'Faceoff': 'FAC',
    'Giveaway': 'GIVE', 
    'Shot': 'SHOT', 
    'Stoppage': 'STOP', 
    'Takeaway': 'TAKE', 
    'Hit': 'HIT', 
    'Missed Shot': 'MISS',
    'Penalty': 'PENL', 
    'Blocked Shot': 'BLOCK', 
    'Goal': 'GOAL', 
    'Period End': 'PEND', 
    'Period Official': None,
    'Shootout Complete': 'SOC', 
    'Game End': 'GEND', 
    'Game Official': 'GOFF',
    'Official Challenge': 'CHL', 
    'Early Intermission Start': 'EISTR',
    'Early Intermission End': 'EIEND', 
    'Emergency Goaltender': 'EGT'
}
# Track whether the code marks a shot or some other event.
# Shots are coded by the categories as 'Goal', 'Missed Shot', 'Shot', and 'Blocked Shot'
SHOT_EVENTS = [ 'SHOT', 'BLOCK', 'GOAL', 'MISS']
# Faceoffs are used as proxies for stoppages, since a faceoff is always used to restart play after a stoppage.
FACEOFF_EVENTS = [ 'FAC' ]
# Mark positions as Forwards/Defense/Goaltender. The positions Center, Left, and Right Wing are all forwards. The generic
# position Forward found in some play-by-plays is also a forward.
FWD_DEF_MAPPING = { 'C': 'FWD', 'L': 'FWD', 'R': 'FWD', 'F': 'FWD', 'D': 'DEF', 'G': 'GOAL'}
# Mark position as skater (synonymously attacker) or goaltender. All positions other than goaltender are considered 
# skater positions.
SKATER_MAPPING = { 'C': 'SKTR', 'L': 'SKTR', 'R': 'SKTR', 'F': 'SKTR', 'D': 'SKTR', 'G': 'GOAL'}
#%% Process Schedules
def get_schedule_local_path(season):
    '''
    Obtains a path object for the local file storing the schedule information for a season.

    Parameters
    ----------
    season : str
        The season for the schedule. Example: '20182019' for the 2018-19 season.

    Returns
    -------
    schedule_path : pathlib.Path
        Path object for the local schedule file, if it exists. If it doesn't exist, points
        to the location that it would exist, allowing saving at that location.

    '''
    current_dir = Path.cwd()
    relative_path = DATA_FOLDER + 'schedule_' + season + '.json'
    schedule_path = current_dir.joinpath(relative_path)
    return schedule_path

def get_schedule_api_url(season):
    '''
    Builds the link to the NHL API schedule endpoint for the requested season.

    Parameters
    ----------
    season : str
        The season for the schedule. Example: '20182019' for the 2018-19 season.

    Returns
    -------
    str
        URL giving the API endpoint to obtain the season schedule. The link is constructed to restrict games
        to the regular season and playoffs.

    '''
    # At the time of writing, querying the website for the 2016-17 season returned a status code of 500.
    # The workaround is to query every calendar date contained in the schedule (2016-10-12 until 2017-06-11).
    
    if (season != '20162017'):
        # Game type of 'R' represents regular season, 'P' represents playoffs. These are the only games for which
        # statistics are officially counted.
        return API_ROOT_URL + '/api/v1/schedule?' + 'season=' + season + '&gameType=R,P'
    else:
        return 'https://statsapi.web.nhl.com/api/v1/schedule?startDate=2016-10-12&endDate=2017-06-11&gameType=R,P'
        
def extract_season_game_feed_links(season):
    '''
    Extracts the live feed links from the schedule returned by the API

    Parameters
    ----------
    season : str
        The season for the schedule. Example: '20182019' for the 2018-19 season.

    Returns
    -------
    schedule_links : list of str
        List of links to live feeds of games for the season.
        Returns None if the API request fails.

    '''
    # api_url is restricted to regular season and playoff games by get_schedule_api_link.
    api_url = get_schedule_api_url(season)
    api_request = requests.get(api_url)
    if (api_request.status_code == 200):
        season = api_request.json()
        logging.info('Success downloading ' + season + 'schedule')
        # The json returned by the API provides a list of calendar dates under the key 'dates'. Each calendar date in
        # turn provides a list of games for that date, keyed by 'games'. Finally, each game provides the live feed link.
        schedule_links = [ game['link'] 
                      for game_date in season['dates'] 
                      for game in game_date['games'] ]    
        return schedule_links
    else:
        logging.error('Error downloading ' + season + 'schedule (Status: ' + str(api_request.status_code)+')')
        return None

def read_game_feed_links(season):
    '''
    Reads the local file containing live feed links for the current season, if it exists.

    Parameters
    ----------
    season : str
        The season for the schedule. Example: '20182019' for the 2018-19 season.

    Returns
    -------
    game_feed_links : list of str
        List of links to live feeds of games for the season, if the local file exists.
        Returns None if the file doesn't exist.
        An individual link has the form '/api/v1/game/2018020256/feed/live', where the substring '2018020256' represents 
        the NHL's game ID for the game. 
    '''
    game_feed_link_path = get_schedule_local_path(season)
    if game_feed_link_path.exists():
        logging.info('Reading ' + season + ' schedule.')
        with game_feed_link_path.open('r') as infile:
            game_feed_links = json.load(infile)
        return game_feed_links
    else:
        return None
    
def get_season_game_feed_links(season, refresh = None):
    '''
    Obtains the live feed links for games in the requested season.

    Parameters
    ----------
    season : str
        The season for the schedule. Example: '20182019' for the 2018-19 season.
    refresh : str, optional
        If not None, ignores the existence of any local files and re-downloads and processes
        the data from the API. This will result in overwriting any current saves. The default is None.

    Returns
    -------
    list of str
        List of links to live feeds of games for the season. An individual link has the form 
        '/api/v1/game/2018020256/feed/live', where the substring '2018020256' represents the NHL's game ID
        for the game. 

    '''
    # Read the data if it exists and no request to refresh/reconstruct the data was sent.
    read_from_file = read_game_feed_links(season) if refresh is None else None
    if read_from_file is None:
        game_feed_links = extract_season_game_feed_links(season)
        # Save the data before returing.
        live_feed_path = get_schedule_local_path(season)
        live_feed_path.touch()
        with live_feed_path.open('w') as outfile:
            json.dump(game_feed_links, outfile)
        return game_feed_links
    else:
        return read_from_file
    
def get_game_feed_links(seasons, refresh=None):
    '''
    Obtains game feed links for every season indicated in seasons.

    Parameters
    ----------
    seasons : str or list
        If a string, should be the season for the schedule. Example: '20182019' for the 2018-19 season.
        If a list, should be a list of strings in the format listed above. Example: ['20182019', '20152016']
        will obtain the information for the 2015-16 and 2018-19 seasons.
    refresh : TYPE, optional
        If a list, should be a list of strings giving seasons in the format above. In this case, passes an
        argument of True to get_season_game_feed_links for any string in seasons that is also in refresh.
        If anything else other than None, passes True to get_season_game_feed_links for all strings in seasons. 
        The default is None.

    Returns
    -------
    List of str
        List of links to live feeds of games for the season. An individual link has the form 
        '/api/v1/game/2018020256/feed/live', where the substring '2018020256' represents the NHL's game ID
        for the game. 

    '''
    if (type(seasons)==str):
        # Assume that a string argument refers to a single season, so just return that season.
        return get_season_game_feed_links(seasons, refresh)
    else:
        # Assume any other argument is a list of seasons.
        if (type(refresh)==list):
            return [ link 
                    for season in seasons 
                    for link in get_season_game_feed_links(season, season if season in refresh else None)]
        else:
            return [ link 
                    for season in seasons 
                    for link in get_season_game_feed_links(season, refresh)]            
 
#%% Download, store, and retrieve raw live feed files

def extract_id_from_live_feed_link(live_feed_link):
    '''
    Extracts the ten-character game id from the live_feed_link.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    str
        The ten-character game id from the live_feed_link. Example: '/api/v1/game/2018020240/feed/live' will return
        '2018020240'

    '''
    return live_feed_link[13:23]
    
def get_live_feed_path(live_feed_link):
    '''
    Obtains the handle for the local version of the live feed file.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    live_feed_path : pathlib.Path
        Path object for the local live feed file, if it exists. If it doesn't exist, points
        to the location that it would exist, allowing saving at that location.

    '''
    current_dir = Path.cwd()
    relative_path = RAW_LIVE_FEED_FOLDER + 'livefeed_' + extract_id_from_live_feed_link(live_feed_link) + '.json'
    live_feed_path = current_dir.joinpath(relative_path)
    return live_feed_path
 
def read_live_feed_local(live_feed_link):
    '''
    Reads the local copy of the live feed for the given link, if it exists.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    live_feed : dict
        Representation of the json object corresponding to the link, if it is saved locally.
        Otherwise returns None.

    '''
    feed_path = get_live_feed_path(live_feed_link)
    if feed_path.exists():
        logging.info('Reading raw feed ' + live_feed_link)
        with feed_path.open('r') as infile:
            live_feed = json.load(infile)
            
        return live_feed
    else:
        return None

def download_live_feed(live_feed_link):
    '''
    Downloads the live feed for the given link from the API.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    dict
        Representation of the json object corresponding to the link, if it exists.
        Otherwise returns None.

    '''
    api_request = requests.get(API_ROOT_URL + live_feed_link)
    
    if (api_request.status_code == 200):
        logging.info('Success downloading raw feed ' + live_feed_link)
        return api_request.json()
    else:
        logging.error('Error downloading raw feed ' + live_feed_link +' (Status: ' + str(api_request.status_code)+')')
        return None

def get_live_feed(live_feed_link, refresh=False):
    '''
    
    Obtains the raw live feed for the link.
    
    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.
    refresh : bool, optional
        If True, ignores the existence of any local files and re-downloads and processes
        the data from the API. This will result in overwriting any current saves. The default is False.

    Returns
    -------
    dict
        Dictionary with the full raw live feed for the link.

    '''

    # Read the file if it already exists locally and there is no request to re-download.
    read_from_file = read_live_feed_local(live_feed_link) if not refresh else None
    if read_from_file is None:
        live_feed = download_live_feed(live_feed_link)
        # Once the raw data is downloaded, save it for faster future processing.
        if live_feed is not None:
           live_feed_path = get_live_feed_path(live_feed_link)
           # Make sure that the folder exists.
           live_feed_path.parent.resolve().mkdir(parents=True, exist_ok=True)
           live_feed_path.touch()
           with live_feed_path.open('w') as outfile:
               json.dump(live_feed, outfile)
        return live_feed
    else:
        return read_from_file    
    pass

#%% Process live feed files into data frame, store, and retrieve data frames.
def convert_to_seconds(time_str, period=None):
    '''
    Converts a time string to seconds.
    
    Parameters
    ----------
    time_str : str
        The time, given in the form 'mm:ss'.
    period : TYPE, optional
        The period of the event. If given, provides the number of seconds since the start
        of the game. The default is None.

    Returns
    -------
    int
        The time of the timestamp in seconds.

    '''
    m, s = time_str.split(':')
    period_adj = 20*60*(period - 1) if period is not None else 0
    return int(m) * 60 + int(s) + period_adj

def parse_live_feed(feed):
    '''
    Parses game live feed to produce a pandas data frame.

    Parameters
    ----------
    feed : dict
        Dictionary containing the live feed data for a game.

    Returns
    -------
    Pandas DataFrame
        Date Frame containing event data from the live feed file. 

    '''
            
    return pd.DataFrame({
        # Game metadata
        'game_id': str(feed['gameData']['game']['pk']),
        'season': str(feed['gameData']['game']['season']),
        'type': feed['gameData']['game']['type'],
        'game_time': feed['gameData']['datetime']['dateTime'],
        'away_code': feed['gameData']['teams']['away']['triCode'] \
            if 'triCode' in feed['gameData']['teams']['away'].keys() else feed['gameData']['teams']['away']['teamName'],
        'home_code': feed['gameData']['teams']['home']['triCode'] \
            if 'triCode' in feed['gameData']['teams']['home'].keys() else feed['gameData']['teams']['home']['teamName'],
        # Venue information. Ideally, we'd just use venue_id, but it is missing for many games, so track venue as well.
        'venue': feed['gameData']['venue']['name'],
        #'venue_id': int(feed['gameData']['venue']['id']) if 'id' in feed['gameData']['venue'].keys() else None,
            
        # Use the event ordering used by the feed.
        'event_idx': [int(play['about']['eventIdx']) for play in feed['liveData']['plays']['allPlays']],
        # Game time of the event.
        # The period and time elapsed are sufficient, but combining these into 'cum_time_elapsed' allows
        # for more succinct determination of time between separate events.
        'period': [int(play['about']['period']) for play in feed['liveData']['plays']['allPlays']],
        # While the ordinal is not crucial, it offers a readable way to determine when the period is a shootout.
        'period_ord': [play['about']['ordinalNum'] for play in feed['liveData']['plays']['allPlays']],
        # Similarly allows easy distinguishing between regulation, overtime, and shootouts.
        'period_type': [play['about']['periodType'] for play in feed['liveData']['plays']['allPlays']],
        'time_elapsed': [play['about']['periodTime'] for play in feed['liveData']['plays']['allPlays']],
        # Calculate the number of seconds into the game of the event.
        'cum_time_elapsed': [ convert_to_seconds(play['about']['periodTime'], int(play['about']['period'])) 
                                        for play in feed['liveData']['plays']['allPlays']],
        # Information about the actual event.
        'event': map(EVENT_TRANSLATION.get, [play['result']['event'] for play in feed['liveData']['plays']['allPlays']]),
        # Track the team corresponding to the event. This will matter for correction of venue bias.
        'event_team_code': [ play['team']['triCode'] \
                            if (('team' in play.keys()) and ('triCode' in play['team'].keys())) else None 
                            for play in feed['liveData']['plays']['allPlays']],
        # Determine whether the event is associated to the home team.
        'event_team_is_home': [(feed['gameData']['teams']['home']['id'] == play['team']['id']) 
                               if ('team' in play.keys()) else None for play in feed['liveData']['plays']['allPlays']],
        # Event coordinates. Note: blocked shots are marked at the location of the block, not the shot.
        'event_coord_x': [float(play['coordinates']['x']) if ('x' in play['coordinates'].keys()) else None 
                          for play in feed['liveData']['plays']['allPlays']],
        'event_coord_y': [float(play['coordinates']['y']) if ('y' in play['coordinates'].keys()) else None 
                          for play in feed['liveData']['plays']['allPlays']],
        # Contains shot type for shots and penalty information for penalties
        'secondary_type': [play['result']['secondaryType'] if ('secondaryType' in play['result'].keys()) else None 
                           for play in feed['liveData']['plays']['allPlays']]       
    })
    
def process_live_feed_frame(frame):
    '''
    Performs post-parsing processing of the live feed data frame.

    Parameters
    ----------
    frame : Pandas DataFrame
        Data frame that has been generated by parsing the live feed for a game.

    Returns
    -------
    frame : Pandas DataFrame
        The input data frame with additional column 'is_rebound' and restricted to only events referring
        to shots.

    '''
    # The main purpose of further processing the frame is to classify shots as to whether they're rebounds or not. 
    # This project follows the convention described in http://blog.war-on-ice.com/annotated-glossary/ that a rebound is 
    # any shot taken within 3 seconds of the previous shot.
    # This is a bit tricky since a shot shouldn't count as a rebound if there was an intervening play stoppage.
    
    # Track whether the event is a shot
    frame['is_shot'] = frame['event'].isin(SHOT_EVENTS)
    # Same concept, but determine whether the event is a stoppage event. While many events stop play, all restarts
    # (except for penalty shots) are done by faceoff. Penalty shots will be dropped later, meaning they are not
    # a concern. As a result, faceoffs are used as proxies for stoppage events.
    frame['is_faceoff'] = frame['event'].isin(FACEOFF_EVENTS) 
        
    # Find the recent shot event, not including the current event.
    # Count the number of previous shot events
    frame['prev_shot_num'] = frame['is_shot'].cumsum() - frame['is_shot']
    # Get the times and indices of shot events.
    shot_times = frame.groupby('prev_shot_num')['cum_time_elapsed'].max()
    shot_indexes = frame.groupby('prev_shot_num')['event_idx'].max()
    # Arbitrarily use -1 for events before the first shot.
    frame['prev_shot_time'] = [shot_times[shot_num - 1] if shot_num > 0 else -1 for shot_num in frame['prev_shot_num']]
    frame['prev_shot_idx'] = [shot_indexes[shot_num - 1] if shot_num > 0 else -1 for shot_num in frame['prev_shot_num']]
    
    # And the most recent faceoff event. Only need the index, not the time, for faceoffs, because the only concern
    # with faceoffs is guaranteeing that there was no intervening faceoff/stoppage between shots. The indexing maintains
    # the order (and allows distinguishing between events occurring less than a second apart).
    frame['prev_faceoff_num'] = frame['is_faceoff'].cumsum() - frame['is_faceoff']
    faceoff_indexes = frame.groupby('prev_faceoff_num')['event_idx'].max()
    frame['prev_faceoff_idx'] = [ faceoff_indexes[fo_num - 1] if fo_num > 0 else -1 for fo_num in frame['prev_faceoff_num'] ]
         
    # Now, rebounds are defined as any shot taken 3 seconds or less after the preceding shot so long as there has
    # been no intervening faceoff.
    frame['is_rebound'] = (frame['cum_time_elapsed'] <= frame['prev_shot_time'] + 3) \
        & (frame['prev_shot_idx'] > frame['prev_faceoff_idx']) & (frame['is_shot'])

    # Limit to shots
    frame = frame[frame['is_shot']].copy()
    
    # Most of the columns generated above are no longer needed
    frame.drop(['is_shot', 'is_faceoff', 'prev_shot_num', 'prev_shot_time', 'prev_shot_idx', 'prev_faceoff_num', 
                'prev_faceoff_idx', 'event_idx'], axis=1, inplace=True)
    
   
    return frame
    
    
def construct_game_live_feed_frame(live_feed_link, refresh=False):
    '''
    Uses the live feed to create a data frame for the information in the feed, if it exists.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.
    refresh : bool, optional
        If True, ignores the existence of any local files and re-downloads and processes
        the data from the API. This will result in overwriting any current saves. The default is False.

    Returns
    -------
    frame : pandas DataFrame
        Data frame representing the game if the live feed exists.
        Returns None otherwise.

    '''
    feed = get_live_feed(live_feed_link, refresh)
    
    if feed is not None:
        # There are two key steps to producing the frame.
        # First, parse the frame and pull out necessary data.
        frame = parse_live_feed(feed)
        # Second, process the frame and add derived columns    
        return process_live_feed_frame(frame)
    else:
        return None

def get_game_live_feed_frame_path(live_feed_link):
    '''
    Constructs the Path object giving a handle to the game live feed data frame.
    This makes no guarantees on whether the file actually exists - the object will refer to the correct file if and only if
    the file already exists.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    frame_path : pathlib.Path
        Path object for the local live feed data frame file, if it exists. If it doesn't exist, points
        to the location that it would exist, allowing saving at that location.

    '''
    current_dir = Path.cwd()
    # Uses live feed formatting for the game id for compactness.
    game_id = extract_id_from_live_feed_link(live_feed_link)
    relative_path = GAME_FRAME_FOLDER + 'livefeed_' + game_id + '.pkl'
    frame_path = current_dir.joinpath(relative_path)
    return frame_path

def read_game_live_feed_frame(live_feed_link):
    '''
    Reads the Pandas live feed data frame for the requested game, if it exists.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    game_frame : andas DataFrame
        Data frame representing the game if the file is saved locally.
        Returns None otherwise.
    '''

    file = get_game_live_feed_frame_path(live_feed_link)
    if file.exists():
        logging.info('Reading live feed data frame for ' + extract_id_from_live_feed_link(live_feed_link))
        game_frame = pd.read_pickle(str(file))
        return game_frame
    else:
        return None
    
def get_game_live_feed_frame(live_feed_link, refresh=False, refresh_frame=False):
    '''
    Obtains a Pandas data frame corresponding to the game live feed for the given link.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.
    refresh : bool, optional
        If True, ignores the existence of any local files and re-downloads and processes
        the data from the API. This will result in overwriting any current saves. The default is False.
    refresh_frame : bool, optional
        Similar to refresh, but only refreshes the data frame. Any locally-saved raw data is kept. Ignored if
        refresh is True. The default is False.

    Returns
    -------
    Pandas data frame
        Data frame representing the live feed data.

    '''   
    refresh_any = refresh | refresh_frame
    read_from_file = read_game_live_feed_frame(live_feed_link) if not refresh_any else None
    if read_from_file is None:
        game_frame = construct_game_live_feed_frame(live_feed_link, refresh)
        # Save the frame
        if game_frame is not None:
            # Make sure that the folder exists.
            game_frame_path = get_game_live_feed_frame_path(live_feed_link)
            game_frame_path.parent.resolve().mkdir(parents=True, exist_ok=True)  
            # Now the file can be saved.              
            game_frame.to_pickle(str(game_frame_path)) 
            
        return game_frame
    else:
        return read_from_file
#%% Download, store, and retrieve raw play-by-play html reports
def get_game_html_report_frame_path(live_feed_link):
    '''
    Obtains a handle to the Pandas data frame created from the html report for the game corresponding to the given link.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    frame_path : pathlib.Path
        Path object for the local live feed data frame file, if it exists. If it doesn't exist, points
        to the location that it would exist, allowing saving at that location.

    '''
    current_dir = Path.cwd()
    # Uses live feed formatting for the game id for compactness.
    game_id = extract_id_from_live_feed_link(live_feed_link)
    relative_path = GAME_FRAME_FOLDER + 'htmlreport_' + game_id + '.pkl'
    frame_path = current_dir.joinpath(relative_path)
    return frame_path

def extract_season_from_link(live_feed_link):
    '''
    Constructs the season id from the live feed link.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    season : str
        Eight-character string identifying the season corresponding to the link. Example: 
        '/api/v1/game/2018020240/feed/live' will return '20182019'

    ''' 
    # The first part of the season string is contained as the first four digits of the game id portion of the url,
    # which can be extracted using extract_id_from_live_feed_link.
    season = extract_id_from_live_feed_link(live_feed_link)[:4]
    # The full season string is those four digits (interpreted as a string giving a year) with four digits representing
    # the following year.
    season += str(int(season) + 1)    
    return season

def get_html_report_url(live_feed_link):
    '''
    Converts the live feed link into the url for the html report of the game corresponding to the link.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    str
        URL for the html report on the NHL.com website. Example: 'http://www.nhl.com/scores/htmlreports/2018/PL020240.HTM'
            for the game in the 2018-2019 season with id 020240.

    '''
    season = extract_season_from_link(live_feed_link)
    # The game portion is the last six characters of the id portion of the link.
    # For '/api/v1/game/2018020240/feed/live', it will be '020240'.
    game = extract_id_from_live_feed_link(live_feed_link)[-6:]
    return 'http://www.nhl.com/scores/htmlreports/' + season + '/PL' + game + '.HTM'

def download_game_html_report(live_feed_link):
    '''
    Downloads the html report for the game corresponding to the given link, if it exists.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    BeautifulSoup
        BeautifulSoup representation of the html report corresponding to the link, if it exists.
        Otherwise returns None.

    '''
    html_report_url = get_html_report_url(live_feed_link)
    report = requests.get(html_report_url)
    if (report.status_code == 200):
        logging.info('Success reading html report ' + html_report_url)
        return BeautifulSoup(report.content, 'html.parser')
    else:
        logging.error('Failure reading html report ' + html_report_url + ' (status: ' + str(report.status_code) +')')
        return None


def get_game_html_report_path(live_feed_link):
    '''
    Obtains the handle for the local version of the html report file.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    live_feed_path : pathlib.Path
        Path object for the local live feed file, if it exists. If it doesn't exist, points
        to the location that it would exist, allowing saving at that location.

    '''
    current_dir = Path.cwd()
    relative_path = RAW_HTML_REPORT_FOLDER + 'htmlreport_' + extract_id_from_live_feed_link(live_feed_link) + '.pkl'
    live_feed_path = current_dir.joinpath(relative_path)
    return live_feed_path

def read_game_html_report(live_feed_link):
    '''
    Reads the local copy of the html report for the game corresponding to the given link, if it exists.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    report_soup : BeautifulSoup
        BeautifulSoup representation of the html report corresponding to the link, if it is saved locally.
        Otherwise returns None.

    '''
    html_report_path = get_game_html_report_path(live_feed_link)
    if html_report_path.exists():
        logging.info('Reading raw html report ' + live_feed_link)
        with html_report_path.open('rb') as infile:
            html_report = pickle.load(infile)
        report_soup = BeautifulSoup(html_report, 'lxml')
        return report_soup
    else:
        return None

def get_game_html_report(live_feed_link, refresh = False):
    '''
    Obtains the BeautifulSoup object for the raw html report data for the game corresponding to the live_feed_link.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.
    refresh : bool, optional
        If True, ignores the existence of any local files and re-downloads and processes
        the data from the API. This will result in overwriting any current saves. The default is False.

    Returns
    -------
    BeautifulSoup
        BeautifulSoup object representing the html report for the game.

    '''
    read_from_file = read_game_html_report(live_feed_link) if not refresh else None
    if read_from_file is None:
        html_report = download_game_html_report(live_feed_link)
         # Save the report
        if html_report is not None:
            # There are currently issues saving the HTML reports locally. Ignore that for the time being.
            pass
            # Make sure that the folder exists.
            html_report_path = get_game_html_report_path(live_feed_link)
            html_report_path.parent.resolve().mkdir(parents=True, exist_ok=True)  
            # Now the file can be saved. 
            with html_report_path.open('wb') as outfile:
                pickle.dump(str(html_report), outfile)
        return html_report
    else:
        return read_from_file

#%% Process html play-by-play reports into data frame, store, and retrieve data frames.
def parse_row_index(row):
    '''
    Extracts the row index from a game event.

    Parameters
    ----------
    row : BeautifulSoup
        BeautifulSoup object referring to a single event

    Returns
    -------
    int
        Index of the game event.

    '''
    return int(list(row.children)[1].get_text())

def parse_row_period(row):
    '''
    Extracts the period from a game event.

    Parameters
    ----------
    row : BeautifulSoup
        BeautifulSoup object referring to a single event

    Returns
    -------
    int
        Period of the event.

    '''
    return int(list(row.children)[3].get_text())

def parse_row_strength(row):
    '''
    Extracts the game strength from the point of view of the team owning the event.

    Parameters
    ----------
    row : BeautifulSoup
        BeautifulSoup object referring to a single event

    Returns
    -------
    str
        String of 'EV' for even strength, 'PP' for power-play, or 'SH' for short-handed.

    '''
    # The strength can contain non-breaking spaces, which are replaced with normal spaces.
    return list(row.children)[5].get_text().replace('\xa0',' ')

def parse_row_time(row, elapsed):
    '''
    Extracts the game time of the event.

    Parameters
    ----------
    row : BeautifulSoup
        BeautifulSoup object referring to a single event
    elapsed : bool
        If True, extracts the elapsed time in the current period. Otherwise, extracts the remaining
        time.

    Returns
    -------
    str
        A time in the form 'mm:ss.

    '''
    # The times are missing initial '0's, meaning they don't match with the live feeds without adjustment.
    # The elapsed time is in index 0, the time remaining is in index 2.
    idx = 2 - 2 * elapsed
    return list(list(row.children)[7].children)[idx].rjust(5, '0')

def parse_row_event(row):
    '''
    Extracts the event-type from the event.

    Parameters
    ----------
    row : BeautifulSoup
        BeautifulSoup object referring to a single event

    Returns
    -------
    str
        Short code denoting the type of event.

    '''
    return list(row.children)[9].get_text()

def is_zone_field(part):
    '''
    Determines if the text field marks an ice zone.

    Parameters
    ----------
    part : str
        Text field extracted from the description.

    Returns
    -------
    Bool
        True if the field marks a zone of the ice (Offensive/Defensive/Neutral), false otherwise.

    '''
    return bool(re.search(r'[Zz]one$', part))

def is_name_field(part):
    '''
    Determines if the text field marks a player name

    Parameters
    ----------
    part : str
        Text field extracted from the description.

    Returns
    -------
    bool
        True is the field marks a player name (determined as it includes the marker \# to indicate a player number is
                                               also in the field).

    '''
    return bool(re.search(r'\#', part))

def is_assist_field(part):
    '''
    Determines if the text field includes assist descriptions for a goal
    Parameters
    ----------
    part : str
        Text field extracted from the description.

    Returns
    -------
    bool
        True if the field describes assists (determined by the presence of the word 'Assist'). False otherwise.

    '''
    return bool(re.search(r'Assist', part))
                
def is_dist_field(part):
    '''
    Determines if the text field gives a distance

    Parameters
    ----------
    part : str
        Text field extracted from the description.

    Returns
    -------
    bool
        True if the field describes a distance (determined by presence of the unit 'ft.'). False otherwise.

    '''
    return bool(re.search(r'ft\.$', part))

def is_shot_type_field(part):
    '''
    Determines if the text field describes a shot type.

    Parameters
    ----------
    part : str
        Text field extracted from the description.

    Returns
    -------
    bool
        True if the field gives a shot type, which is one of 'Slap shot', 'Snap shot', 'Backhand', 'Wrist', 'Deflected', 'Tip-in', or
        'Wraparound'.

    '''
    return bool(re.search(r'S[nl]ap|Backhand|Wrist|Deflected|Tip|Wrap', part))

def is_miss_type_field(part):
    '''
    Determines if the text field describes how a shot missed.

    Parameters
    ----------
    part : str
        Text field extracted from the description.

    Returns
    -------
    bool
        True if the field describes how a shot missed. This could be 'Wide of Net', 'Over Net', 'Hit Goalpost', 'Hit Crossbar'.

    '''
    return bool(re.search(r'Net|Goalpost|Crossbar', part))

def extract_distance(part):
    '''
    Extracts the distance of a shot from a text field.

    Parameters
    ----------
    part : TYPE
        Text field extracted from the description. Should represent a distance in the form 'NNN ft.'

    Returns
    -------
    int
        The integer value (in feet) of the shot distance. Returns None if no unit marker was found.

    '''
    res = re.search('\d+(?=.*ft\.)', part)
    if res:
        return int(res.group(0))
    else:
        return None

def parse_row_desc_components(parts, event):
    '''
    Extracts the shot distance, event zone, shot type, and a description of why the shot missed (for misses) from
    the description describing a shot.

    Parameters
    ----------
    parts : list of str
        substrings of the description, split at ','
    event : str
        A string indicating the type of event that the description field describes. Can be 'SHOT', 'GOAL', 'MISS',
        or 'BLOCK'.

    Returns
    -------
    dict
        Dictionary containing which ever of the distance, zone, shot, and miss type are given by the description.

    '''
    
# Generally, the order of the fields are the following.
# 1. Player (usually the shooter, but for event 'BLOCK', gives the player who blocked the shot as well).
# 2. Shot-type. Usually (but not always) present.
# 3. How shot was missed. Only present for event 'MISS', but this will also be used to code blocks (event 'BLOCK') and 
#   saves (event 'SHOT') as well.
# 4. Event zone.
# 5. Shot distance. Not present for 'BLOCK' event.
# 6. Assists. Only present if assists were credited for 'GOAL' event. Not present for any other event type.

# Reverse the order of the fields. If the fields are present, they will always have the same relative order.
    parts = [part for part in reversed(parts)]
    idx = 0
    parsed = { 'shot_dist': None, 'event_zone': None, 
              'miss_type': None, 'shot_type': None }
    
    # Check for assists, which only matters for goals.
    if ((idx < len(parts)) & (event=='GOAL')):
        if (is_assist_field(parts[idx])):
            # Always ignore the assist field
            idx += 1
    
    # Next, check for a distance field
    if ((idx < len(parts)) & is_dist_field(parts[idx])):
        dist = extract_distance(parts[idx])
        parsed['shot_dist'] = dist
        idx += 1
        
    # Next, check for a zone field
    if ((idx < len(parts)) & is_zone_field(parts[idx])):
        parsed['event_zone'] = parts[idx]
        idx += 1

    # Next is the miss-type field, only matters if event is MISS, although blocks and saved-shots will be coded here
    # as well.
    if (idx < len(parts)):
        if (event=='MISS'):
            if (is_miss_type_field(parts[idx])):
                # If it's an obvious miss, then include it.
                parsed['miss_type'] = parts[idx]
                idx += 1
            else:
                # Even if it's not an obvious miss, presumptively count it as a
                # miss in case the field includes an unusual description or incorrect spelling.
                # The only evidence against being a miss-type is evidence for the field being a name field
                # or a shot-type field.
                guess_false = is_name_field(parts[idx]) | is_shot_type_field(parts[idx])
                if not guess_false:
                    # There is no good evidence against it being a miss type.
                    parsed['miss_type'] = parts[idx]
                    idx += 1     
        elif (event=='BLOCK'):
            parsed['miss_type'] = 'Block'
        elif (event=='SHOT'):
            parsed['miss_type'] = 'Save'
           
           
    # Final field to check is presumptively the shot-type field.
    if (idx < len(parts)):
        if (is_shot_type_field(parts[idx])):
            # Obvious shot
            parsed['shot_type'] = parts[idx]
        else:
            # Like with miss-type, this field is presumptively the shot-type, unless it is obviously a name field.
            if not is_name_field(parts[idx]):
                parsed['shot_type'] = parts[idx]
                
       
    return parsed

def parse_row_desc(row):
    '''
    Extracts the description from an event and splits it for ease of future parsing.

    Parameters
    ----------
    row : BeautifulSoup
        BeautifulSoup object referring to a single event

    Returns
    -------
    list of str
        The event description split at commas. For goals, an additional split is added after the 
        distance and before the assists.

    '''
    # The description can contain non-breaking spaces, which are replaced with normal spaces.
    # Goal descriptions omit a comma between the shot distance and the assists. Since the units are always 'ft.' and will
    # end up being dropped later, they're replaced with ',' here to aid with the string splitting and parsing.
    # Finally, the description is split to make it easier to parse at the next step.
    
    # Moving to increase robustness. T
    # The order of items depends on the type of row being parsed.
    
    event = parse_row_event(row)
    desc = list(row.children)[11].get_text(separator=', ')

                                                                                            
    if (event in SHOT_EVENTS):
        desc_parts = desc.split(', ')
        return parse_row_desc_components(desc_parts, event)
    else:
        return desc


def parse_on_ice_pos(row, home):
    '''
    Extracts the positions of the players on ice for one of the teams

    Parameters
    ----------
    row : BeautifulSoup
        BeautifulSoup object referring to a single event
    home : bool
        If True, extract the home team. Otherwise, extract the away team.

    Returns
    -------
    Counter
        Counter of all positions found on the ice for the event. Example: if there are two centers, a left winger,
        two defense and a goaltender, will return Counter({'C': 2, 'L': 1, 'D': 2, 'G': 1})

    '''
    # The visiting team uses index 13, the home team 15. Start by grabbing the correct section of the row.
    idx = 13 + 2 * home
    sec = list(row.children)[idx].find('table')
    if sec is not None:
        # If the section exists, we can break it down further into subsections for each player
        player_list = sec.find_all('table')
        # Each player subsection contains two cells. The player position is the text from the second cell.
        return Counter([ player.find_all('td')[1].get_text() for player in player_list])
    else:
        return None

def parse_penalty_shot(row):
    '''
    Determines whether the event was a penalty shot.

    Parameters
    ----------
    row : BeautifulSoup
        BeautifulSoup object referring to a single event

    Returns
    -------
    bool
        True if the description indicates the shot was a penalty shot, False otherwise.

    '''
    # Search the description for use of the term 'Penalty Shot'.
    return bool(re.search('Penalty Shot', list(row.children)[11].get_text().replace('\xa0',' ')))
 
    
def parse_game_html_report(report):
    '''
    Parses game html report to produce a pandas data frame.

    Parameters
    ----------
    report : BeautifulSoup
        BeautifulSoup object for the html report page.

    Returns
    -------
    frame :     Pandas DataFrame
        Date Frame containing event data from the html report. 

    '''
    # Play-by-play rows are either all the same class or one of two classes.
    event_rows = report.find_all('tr', class_ = re.compile("(evenColor|oddColor)"))
    
    # Row children are
    #   1: Index
    #   3: Period (In regular season, OT is 4 and SO is 5)
    #   5: Strength (Even strength = EV, Power play = PP, Short-handed = SH)
    #   7: Time elapse / Time remaining
    #   9: Event type
    #  11: Event detailed description
    #  13: Visiting/away players on ice / jersey numbers and positions
    #  15: Home players on ice / jersey numbers and positions    
    frame = pd.DataFrame({
        # Include metadata because we need to investigate why frames aren't matching.
        #'idx': [ parse_row_index(row) for row in event_rows],
        'period': [ parse_row_period(row) for row in event_rows],
        'strength': [ parse_row_strength(row) for row in event_rows],
        'time_elapsed': [ parse_row_time(row, True) for row in event_rows],
        'time_remaining': [ parse_row_time(row, False) for row in event_rows],
        'event': [ parse_row_event(row) for row in event_rows],
        'desc': [ parse_row_desc(row) for row in event_rows],
        'is_penalty_shot': [ parse_penalty_shot(row) for row in event_rows],
        # Positions for players on ice, away and home respectively
        'pos_a':  [ parse_on_ice_pos(row, False) for row in event_rows],
        'pos_h': [ parse_on_ice_pos(row, True) for row in event_rows]
    })    
    return frame

def process_parsed_report(frame):
    '''
    Performs post-parsing processing of the html report data frame.

    Parameters
    ----------
    frame : pandas data frame
        Data frame that has been generated by parsing the html report for a game.

    Returns
    -------
    frame : Pandas DataFrame
        The input data frame with additional columns 'event_zone', 'how_missed', 'shot_dist', and 'shot_type'.
        All non-shot events are also removed.

    '''
    frame['seconds_remaining'] = frame['time_remaining'].apply(lambda x: convert_to_seconds(x))
    
    #FWD_DEF_MAPPING = { 'C': 'FWD', 'L': 'FWD', 'R': 'FWD', 'F': 'FWD', 'D': 'DEF', 'G': 'GOAL'}
    #SKATER_MAPPING = { 'C': 'SKTR', 'L': 'SKTR', 'R': 'SKTR', 'F': 'SKTR', 'D': 'SKTR', 'G': 'GOAL'}

    frame['fwd_def_a'] = frame['pos_a'].apply(lambda x: 
                                              Counter(map(FWD_DEF_MAPPING.get, x.elements())) if x is not None else None)
    frame['fwd_def_h'] = frame['pos_h'].apply(lambda x: 
                                              Counter(map(FWD_DEF_MAPPING.get, x.elements())) if x is not None else None)
    frame['skaters_a'] = frame['pos_a'].apply(lambda x: 
                                              Counter(map(SKATER_MAPPING.get, x.elements())) if x is not None else None)
    frame['skaters_h'] = frame['pos_h'].apply(lambda x: 
                                              Counter(map(SKATER_MAPPING.get, x.elements())) if x is not None else None)
                                                                                                                                          
    frame['goalie_pulled_a'] = frame['pos_a'].apply(lambda x: x['G']==0 if x is not None else None)
    frame['goalie_pulled_h'] = frame['pos_h'].apply(lambda x: x['G']==0 if x is not None else None)
    
    expanded_dicts = frame['desc'].apply(pd.Series)
    frame['shot_dist'] = expanded_dicts['shot_dist']
    frame['event_zone'] = expanded_dicts['event_zone']
    frame['miss_type'] = expanded_dicts['miss_type']
    frame['shot_type'] = expanded_dicts['shot_type']
    
    # Finally, reduce the table to only shot events.
    frame = frame[frame['event'].isin(SHOT_EVENTS)].copy()    
    
    # Eliminate description, as it is no longer needed.
    frame.drop('desc', inplace=True, axis=1)
    return frame
    
def construct_game_html_report_frame(live_feed_link, refresh = False):
    '''
    Constructs a Pandas data frame from the html report data for the requested game.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.
    refresh : bool, optional
        If True, ignores the existence of any local files and re-downloads and processes
        the data from the API. This will result in overwriting any current saves. The default is False.

    Returns
    -------
    Pandas data frame
        Pandas data frame representing the game if the html report exists.
        Returns None otherwise.

    '''
    report = get_game_html_report(live_feed_link, refresh)
    if report is None:
        return None
    else:
        frame = parse_game_html_report(report)
        frame['game_id'] = extract_id_from_live_feed_link(live_feed_link)
        return process_parsed_report(frame)
    
def read_game_html_report_frame(live_feed_link):
    '''
    Reads the Pandas html report data frame for the requested game, if it exists locally.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    game_frame : Pandas data frame
        Data frame representing the html report data, if it exists.
        Returns None otherwise.

    '''
    frame_path = get_game_html_report_frame_path(live_feed_link)
    if frame_path.exists():
        logging.info('Reading html frame ' + live_feed_link)
        game_frame = pd.read_pickle(str(frame_path))
        return game_frame
    else:
        return None
    
def get_game_html_report_frame(live_feed_link, refresh = False, refresh_frame=False):
    '''
    Obtains a Pandas data frame corresponding to the HTML report for the game corresponding to the
    live feed link.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.
    refresh : bool, optional
        If True, ignores the existence of any local files and re-downloads and processes
        the data from the API. This will result in overwriting any current saves. The default is False.
    refresh_frame : bool, optional
        Similar to refresh, but only refreshes the data frame. Any locally-saved raw data is kept. Ignored if
        refresh is True. The default is False.

    Returns
    -------
    Pandas data frame
        Data frame representing the html report data.

    '''
    refresh_any = refresh | refresh_frame
    read_from_file = read_game_html_report_frame(live_feed_link) if not refresh_any else None
    if read_from_file is None:
        game_frame = construct_game_html_report_frame(live_feed_link, refresh)
         # Save the frame
        if game_frame is not None:
            # Make sure that the folder exists.
            game_frame_path = get_game_html_report_frame_path(live_feed_link)
            game_frame_path.parent.resolve().mkdir(parents=True, exist_ok=True)  
            # Now the file can be saved.              
            game_frame.to_pickle(str(game_frame_path)) 
        return game_frame
    else:
        return read_from_file
#%% Read and combine partial game frames into a single combined frame for the game.
def combine_frames(live_feed_frame, html_report_frame):
    '''
    Combine the frames created from the game live feed and the html play-by-play report.

    Parameters
    ----------
    live_feed_frame : Pandas data frame
        Data frame obtained from the game live feed.
    html_report_frame : Pandas data frame
        Data frame obtained from the html play-by-play report.

    Returns
    -------
    Pandas data frame
        Data frame describing every shot event in a single game.

    '''
    return pd.merge(live_feed_frame, html_report_frame, how='outer', left_on=['period', 'time_elapsed', 'event'], 
             right_on=['period', 'time_elapsed', 'event'],  suffixes=['_livefeed', '_htmlreport'])
 
def process_combined_frame(combined_frame):
    '''
    Work toward cleaning the combined data frame obtained from combine_frames

    Parameters
    ----------
    combined_frame : Pandas data frame
        Combined data frame obtained from combine_frames.

    Returns
    -------
    combined : Pandas data frame
        Data frame that has been partially cleaned.

    '''
    # Only shots taken during normal gameplay should be included. That means that any penalty shots (special untimed
    # shots which involve one player against a goaltender) should be excluded. Penalty shots are (although rarefy)
    # awarded for penalties which a referee feels prevented a scoring chance (such as a player on a breakaway being
    # tripped). Regular season games that are tied after an overtime period are also decided by a series of penalty
    # shots in a 'Shootout' period.
    # Exclude both types. Penalty shots in regulation should be marked as such from the description.
    # Shootouts will have the period described as 'SO'.
    combined = combined_frame[combined_frame['is_penalty_shot']==False].copy()
    combined = combined[combined['period_ord'] != 'SO'].copy()
    # The 'is_penalty_shot' column is no longer needed since penalty shots have been removed from the frame.
    combined.drop(['is_penalty_shot'], axis=1, inplace=True)
        
    # Blocks use the blocking team's viewpoint. This should be swapped to use the shooting team's viewpoint.
    # Coordinates don't need to change in the live feed, but event_team_id, event_team_is_home, strength, and event_zone 
    # need to be flipped.
    # Flip event team home flag. This just swaps True and False, however it's important to coerce the type to bool first.
    combined['event_team_is_home'] = combined['event_team_is_home'].astype(bool)
    combined['event_team_is_home'] = np.where(combined['event']!='BLOCK', 
                                              combined['event_team_is_home'], 
                                              ~combined['event_team_is_home'] )
    # Flip team id.
    away_code =  combined['away_code']
    home_code =  combined['home_code']
    combined['event_team_code'] = np.where(combined['event']!='BLOCK', 
                                         combined['event_team_code'], 
                                         np.where(combined['event_team_code']==away_code, home_code, away_code) )
    # Flip zones. Here, it's a simple swap of offensive and defensive zones, with neutral zone left unchanged
    block_zone_swap = { 'Def. Zone': 'Off. Zone', 
                       'Off. Zone': 'Def. Zone'}
    combined['event_zone'] = np.where(combined['event']!='BLOCK', 
                                      combined['event_zone'], 
                                      combined['event_zone'].replace(block_zone_swap) )
    # Strengths are similar to zones. Here, 'EV' (even strength) is left alone, but 'PP' and 'SH' (power-play and 
    # short-handed) are swapped.
    block_strength_swap = { 'PP': 'SH', 
                           'SH': 'PP' }
    combined['strength'] = np.where(combined['event']!='BLOCK', 
                                      combined['strength'], 
                                      combined['strength'].replace(block_strength_swap) )    
    
    # Standardize coordinate directions.

    # Roughly half the shots are charted on each half of the ice. The data will be standardized so that all shots are
    # mapped as occurring toward the same end of the ice. This is arbitrarily chosen to be the goal with positive x-
    # coordinate.
    
    # First, partially standardize to that the home team is always seen to be shooting against the goal with positive
    # x-coordinate and the visiting team toward the other goal.
    # Since it is still possible for a shot to be taken from a location with negative x-coordinate, the home end
    # of the ice is determined first. Teams switch ends of the ice each period. Grouping the signs of shot coordinates
    # by period and home/away allows determination of which end of the ice is being attacked.
    
    combined['home_attacks_positive'] = ((combined['event_coord_x'] >= 0) & combined['event_team_is_home']) | \
                                        ((combined['event_coord_x'] <= 0) & ~combined['event_team_is_home'])
    combined['home_attacks_positive'] = combined['home_attacks_positive'].astype(float)      
                           
    # In most cases, the mean will be within 0.1 of either end. It appears reasonable to assume that any period with mean
    # greater than 0.5 indicates that the home attack end has positive x-coordinates. Additionally, any period with
    # mean less than 0.5 indicates that the home attack end has negative x-coordinates. These periods will be rotated
    # 180 degrees for the first part of the standardization.
    combined['home_end_correct'] = combined.groupby('period')[['home_attacks_positive']].transform('mean')

    # Only coordinates need to change. In this case, since the intent is to rotate 180 degrees, x-coordinates and y-coordinates
    # are both negated for periods when the home team is attacking the negative x-coordinate end.
    
    combined['event_coord_x'] = np.where(combined['home_end_correct'] >= 0.5, 
                                         combined['event_coord_x'], 
                                         -combined['event_coord_x'])
    combined['event_coord_y'] = np.where(combined['home_end_correct'] >= 0.5, 
                                         combined['event_coord_y'], 
                                         -combined['event_coord_y'])
    
    # Finish the standardization. Since the data has been adjusted so that the home team is always attacking a single end,
    # rotating all visiting team shots will produce the desired result.
    combined['event_coord_x'] = np.where(combined['event_team_is_home'] == True, 
                                         combined['event_coord_x'], 
                                         -combined['event_coord_x'])
    combined['event_coord_y'] = np.where(combined['event_team_is_home'] == True, 
                                         combined['event_coord_y'], 
                                         -combined['event_coord_y'])
    
    # The new columns have served their purpose and can be removed.
    combined.drop(['home_end_correct', 'home_attacks_positive'], axis=1, inplace=True)
        
    # Calculate shot distance from coordinates.
    # The distance is calculated to the center (y=0) of the goal line. The goal line is 11 feet from the 
    # end boards, which are 100 feet from center ice, giving x=89.
    
    combined['calc_dist'] = np.sqrt((combined['event_coord_x']-89)**2 + combined['event_coord_y']**2)
    combined['dist_difference'] = np.abs(combined['calc_dist'] - combined['shot_dist'])
    return combined

def construct_combined_frame(live_feed_frame, html_report_frame):
    '''
    Pipeline to merge and process the two frames created from the game live feed and the html play-by-play report.

    Parameters
    ----------
    live_feed_frame : Pandas data frame
        Data frame created from the game live feed.
    html_report_frame : Pandas data frame
        Data frame created from the html play-by-play report.

    Returns
    -------
    Pandas data frame
        Combined frame.

    '''
    # Merge the frames.
    combined = combine_frames(live_feed_frame, html_report_frame)
    combined = process_combined_frame(combined)
    return combined.reset_index(drop=True)

def get_game_combined_frame_path(live_feed_link):

    '''
    Obtains the path to the Pandas data frame created from the html report for the game corresponding to the given link.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    frame_path : pathlib.Path
        Path object for the local live feed data frame file, if it exists. If it doesn't exist, points
        to the location that it would exist, allowing saving at that location.

    '''
    current_dir = Path.cwd()
    # Uses live feed formatting for the game id for compactness.
    game_id = extract_id_from_live_feed_link(live_feed_link)
    relative_path = GAME_FRAME_FOLDER + 'combined_' + game_id + '.pkl'
    frame_path = current_dir.joinpath(relative_path)
    return frame_path
    
def read_game_combined_frame(live_feed_link):
    '''
    Reads the combined frame from local file, if it exists.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.

    Returns
    -------
    game_frame : Pandas data frame
        Data frame combining the live feed and the html report data.

    '''
    
    frame_path = get_game_combined_frame_path(live_feed_link)
    if frame_path.exists():
        logging.info('Reading combined data frame for ' + extract_id_from_live_feed_link(live_feed_link))
        game_frame = pd.read_pickle(str(frame_path))
        return game_frame
    else:
        return None
    
def get_game_combined_frame(live_feed_link, refresh_combine=False, refresh_all=False, refresh_feed=False, 
                            refresh_feed_frame=False, refresh_html=False, refresh_html_frame=False):
    '''
    Obtains the combined data frame for the game corresponding to live_feed_link.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.
    refresh_combine : bool, optional
        Re-combine the frames but otherwise use any locally-saved files. Has no effect if any other option is set.
        The default is False.
    refresh_all : bool, optional
        Forces re-downloading of the live feed and the html report. This option causes all other options to be ignored
        if set. Existing files will be overwritten. The default is False.
    refresh_feed : bool, optional
        Forces re-downloading of the live feed, but not the html report. Existing live feed files will be overwritten.
        This option is ignored if refresh_all is set. The default is False.
    refresh_feed_frame : bool, optional
        Forces re-creation of the live feed frame without re-downloading existing raw files. This option is ignored if
        refresh_feed is set. The default is False.
    refresh_html : bool, optional
        Forces re-downloading of the html report, but not the live feed. Existing html report files will be overwritten.
        This option is ignored if refresh_all is set. The default is False.
    refresh_html_frame : bool, optional
        Forces re-creation of the html report frame without re-downloading existing raw files. This option is ignored if
        refresh_html is set. The default is False.

    Returns
    -------
    Pandas data frame
        Data frame combining the live feed and the html report data.

    '''
    
    # If any of the refresh options are true, the combined local file shouldn't be read as it will need to be 
    # recreated
    refresh_any = refresh_combine | refresh_all | refresh_feed | refresh_feed_frame | refresh_html | refresh_html_frame
    #logging.debug('Here?')
    read_from_file = read_game_combined_frame(live_feed_link) if not refresh_any else None
    #logging.debug('Or here?')
    if read_from_file is None:
        # There are multiple reasons the file may need to be recreated. In the event of refresh_combine, the 
        # constituent frames can simply be read. For refresh_all, everything needs to be re-created.
        # Pass refresh states onto the individual loading functions, with refresh_all overriding everything else if true.
        feed_frame = get_game_live_feed_frame(live_feed_link, refresh_all | refresh_feed, refresh_all | refresh_feed_frame)
        html_frame = get_game_html_report_frame(live_feed_link, refresh_all | refresh_html, refresh_all | refresh_html_frame)
        
        # Combining the frames is a required action. 
        #logging.debug('Do we execute this?')
        combined_frame = construct_combined_frame(feed_frame, html_frame)
        #logging.debug('Or this?')
        
        # If the combination occurred successfully, save the file.
        if combined_frame is not None:
            # Make sure that the folder exists.
            combined_frame_path = get_game_combined_frame_path(live_feed_link)
            combined_frame_path.parent.resolve().mkdir(parents=True, exist_ok=True)
            # The directory exists, so the file can be saved.                
            combined_frame.to_pickle(str(combined_frame_path)) 
        return combined_frame           
            
    else:
        return read_from_file

def retrieve_all(link_list, refresh=False, refresh_feed=False, refresh_html=False):
    '''
    Downloads and locally stores all game live feeds and html reports for the games with links provided

    Parameters
    ----------
    link_list : list of str
        List of API links to the game live feeds.
    refresh : bool, optional
        If true, forces re-download of all files, even if local versions already exist. Overwrites
        any existing files. The default is False.
    refresh_feed : bool, optional
        If true, forces re-download of all game live feed files, even if local versions already exist. Overwrites
        any existing files. Ignored if refresh is True. The default is False.
    refresh_html : bool, optional
        If true, forces re-download of all html play-by-play report files, even if local versions already exist. Overwrites
        any existing files. Ignored if refresh is True. The default is False.

    Returns
    -------
    None.

    '''
    for live_feed_link in link_list:
        get_live_feed(live_feed_link, refresh | refresh_feed)
        get_game_html_report(live_feed_link, refresh | refresh_html)
 
def get_game_combined_frame_from_local(live_feed_link, refresh_combine=False, refresh_all=False, refresh_feed=False, 
                                       refresh_html=False):
    '''
    Gets the combined data frame for the game corresponding to the provided link.

    Parameters
    ----------
    live_feed_link : str
        The live feed link of the game for the frame. Example: '/api/v1/game/2018020240/feed/live' 
        for the game in the 2018-2019 season with id 020240. See the documentation for get_game_feed_links
        for more information.
    refresh_combine : bool, optional
        Re-combine the frames but otherwise use any locally-saved files. Has no effect if any other option is set.
        The default is False.
    refresh_all : bool, optional
        If True, force re-creation of live feed and html report data frames from . The default is False.
    refresh_feed : bool, optional
        Forces re-creation of the live feed frame without re-downloading existing raw files. This option is ignored if
        refresh_all is set. The default is False.
    refresh_html : bool, optional
        Forces re-creation of the html report frame without re-downloading existing raw files. This option is ignored if
        refresh_all is set. The default is False.
        
    Returns
    -------
    Pandas data frame
        Data frame including shots for the game.

    '''
    refresh_feed_frame = refresh_all | refresh_feed
    refresh_html_frame = refresh_all | refresh_html
    return get_game_combined_frame(live_feed_link, refresh_combine=refresh_combine, refresh_feed_frame=refresh_feed_frame, 
                            refresh_html_frame=refresh_html_frame)
#%% Obtain and process data.
def check_live_feeds_for_missing_data(live_feed_links):
    '''
    Checks game live feeds to determine if a play-by-play is included in the feed. Produces a list of live feeds
    with no play by play.

    Parameters
    ----------
    live_feed_links : list of str
        List of live feed links.

    Returns
    -------
    bad_links : list of str
        List of links where the live-feed contained no play-by-play.

    '''
    bad_links = []
    for live_feed_link in live_feed_links:
        feed = get_live_feed(live_feed_link)
        if len(feed['liveData']['plays']['allPlays']) == 0:
            bad_links.append(live_feed_link)
    return bad_links

def get_missing_link_path():
    '''
    Provides path to local file saving live feed links with missing play-by-play

    Returns
    -------
    bad_link_path : Path object
        Path to json file storing live feed links missing play-by-play.

    '''
    current_dir = Path.cwd()
    relative_path = DATA_FOLDER + 'bad_links.json'
    bad_link_path = current_dir.joinpath(relative_path)   
    return bad_link_path

def read_missing_links():
    '''
    Reads local file storing live feed links with missing play-by-play, if it exists.

    Returns
    -------
    bad_links : list of str
        List of links where the live-feed contained no play-by-play.

    '''
    bad_link_path = get_missing_link_path()
    if bad_link_path.exists():
        logging.info('Reading bad links.')
        with bad_link_path.open('r') as infile:
            bad_links = json.load(infile)
        return bad_links
    else:
        return None        

def get_missing_links(live_feed_links):
    '''
    Get local file storing live feed links with missing play-by-play, if it exists. Otherwise, produce and save
    the list of links.

    Parameters
    ----------
    live_feed_links : list of str
        List of live feed links.

    Returns
    -------
    list of str
        List of links where the live-feed contained no play-by-play.

    '''
    read_from_file = read_missing_links()
    if read_from_file is None:
        bad_links = check_live_feeds_for_missing_data(live_feed_links)
        bad_link_path = get_missing_link_path()
        bad_link_path.touch()
        with bad_link_path.open('w') as outfile:
            json.dump(bad_links, outfile)       
        return bad_links
    else:
        return read_from_file
    
# Get all game links from the desired seasons.
game_links = get_game_feed_links(SEASON_LIST)
# Process links where live feed file is missing play-by-play. Ignore these.
missing_links = get_missing_links(game_links)
# There are two games in the list with broken play-by-play in the HTML reports. Ignore these as well.
broken_links = ['/api/v1/game/2010020124/feed/live', '/api/v1/game/2013020971/feed/live']
bad_links = missing_links + broken_links

# Create frames for each game.
combined_frame_list = [get_game_combined_frame_from_local(link) for link in game_links if link not in bad_links]