import json
import pandas as pd
import urllib3
import requests
# I am familiarizing myself with the use of the NBA Stats API from Ryan Davis' tutorial on NBA Data processing
# https://github.com/rd11490/NBA_Tutorials/blob/master/README.md
# This is the first exercise, what he calls "Players on court"
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Need to use headers for NBA API data calls
header_data = {
    'Host': 'stats.nba.com',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36',
    'Referer': 'stats.nba.com',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
}

# These functions create endpoints
def get_pbp_url(id):
    '''
    :param id: game id
    :return: string, url with appended game id
    '''
    return "https://stats.nba.com/stats/playbyplayv2/?gameId={0}&startPeriod=0&endPeriod=14".format(id)

def get_advanced_pbp_url(id, start, end):
    '''
    :param id: game id
    :param start: starting period
    :param end: ending period
    :return: string, url for approporiate API url
    More specific version of the function above
    '''
    return "https://stats.nba.com/stats/boxscoreadvancedv2/?gameId={0}&startPeriod=0&endPeriod=14&startRange={1}&endRange={2}&rangeType=2".format(id, start, end)

# Need to generate the http client
http = urllib3.PoolManager()

# This function will download and extract url data into a dataframe

def extract_data(url):
    '''
    :param url: string, url to extract data from
    :return: dataframe containing that page
    Function by Ryan Davis, as I'm not familiar with urllib
    '''
    r = requests.get(url, headers=header_data)
    resp = r.json()
    results = resp['resultSets'][0]
    headers = results['headers']
    rows = results['rowSet']
    frame = pd.DataFrame(rows)
    frame.columns = headers
    return frame

# Function for calculating start time at every period
def calc_time_at_period(period):
    '''
    :param period: game period
    :return: time which period started
    '''
    if period > 5:
        return (720 * 4 + (period - 5) * (5 * 60)) * 10
    else:
        return (720 * (period - 1)) * 10

# Need something to delineate subs going in and subs going out
def split_subs(frame, tag):
    '''
    :param df: Data frame with API game data
    :param tag: "IN" or "OUT"
    :return: Subbed players of a given type
    '''
    subs = frame[[tag, 'PERIOD', 'EVENTNUM']]
    subs['SUB'] = tag
    subs.columns = ['PLAYER_ID', 'PERIOD', 'EVENTNUM', 'SUB']
    return subs

def frame_to_row(df):
    '''
    :param df: play by play data frame
    :return: list containing all players in game
    '''
    team1 = df['TEAM_ID'].unique()[0]
    team2 = df['TEAM_ID'].unique()[1]
    players1 = df[df['TEAM_ID'] == team1]['PLAYER_ID'].tolist()
    players1.sort()
    players2 = df[df['TEAM_ID'] == team2]['PLAYER_ID'].tolist()
    players2.sort()

    lst = [team1]
    lst.append(players1)
    lst.append(team2)
    lst.append(players2)


    return lst

def get_players_on_court_at_start_of_period_df(game_id):
    '''
    :param game_id: game id to get data
    :return: returns starting players on court in every period
    '''
    # Extract data for given game id
    frame = extract_data(get_pbp_url(game_id))
    print(frame)
    # Filter out as to only include substitutions
    substitutionsOnly = frame[frame["EVENTMSGTYPE"] == 8][['PERIOD', 'EVENTNUM', 'PLAYER1_ID', 'PLAYER2_ID']]
    substitutionsOnly.columns = ['PERIOD', 'EVENTNUM', 'OUT', 'IN']
    # Split in and out subs, and the full data set
    subs_in = split_subs(substitutionsOnly, 'IN')
    subs_out = split_subs(substitutionsOnly, 'OUT')
    full_subs = pd.concat([subs_out, subs_in], axis=0).reset_index()[['PLAYER_ID', 'PERIOD', 'EVENTNUM', 'SUB']]
    # Group data by player and period, then take first substitution in each period
    first_event_of_period = full_subs.loc[full_subs.groupby(by=['PERIOD', 'PLAYER_ID'])['EVENTNUM'].idxmin()]
    # Filter only to players who's first event was being subbed in
    players_subbed_in_at_each_period = first_event_of_period[first_event_of_period['SUB'] == 'IN'][
        ['PLAYER_ID', 'PERIOD', 'SUB']]
    # List of each period in the game
    periods = players_subbed_in_at_each_period['PERIOD'].drop_duplicates().values.tolist()
    # Calculate the start and end time of the period
    # (offset by .5 seconds so that there is no collision at the start/end barrier between periods).
    frames = []
    for period in periods:
        low = calc_time_at_period(period) + 5
        high = calc_time_at_period(period + 1) - 5
        # Then download the boxscore for that time range, extract the player name, id, and team.
        boxscore = get_advanced_pbp_url(game_id, low, high)
        boxscore_players = extract_data(boxscore)[['PLAYER_NAME', 'PLAYER_ID', 'TEAM_ID']]
        boxscore_players['PERIOD'] = period

        players_subbed_in_at_period = players_subbed_in_at_each_period[
            players_subbed_in_at_each_period['PERIOD'] == period]
        # Join the boxscore with the sub frame from above and filter out any rows where the join was successful
        joined_players = pd.merge(boxscore_players, players_subbed_in_at_period, on=['PLAYER_ID', 'PERIOD'], how='left')
        joined_players = joined_players[pd.isnull(joined_players['SUB'])][
            ['PLAYER_NAME', 'PLAYER_ID', 'TEAM_ID', 'PERIOD']]
        frame = frame_to_row(joined_players)
        frame.append(period)
        frames.append(frame)

    players_on_court_at_start_of_period = pd.DataFrame(frames)
    cols = ['TEAM_ID_1', 'TEAM_1_PLAYERS', 'TEAM_ID_2', 'TEAM_2_PLAYERS', 'PERIOD']
    players_on_court_at_start_of_period.columns = cols
    return players_on_court_at_start_of_period

def generate_game_id_list(season, game_number, season_part):
    '''
    :param season: string, the last 2 digits of the year the season started
    :param game_number: int, The number of games played in that season
    :param season_part: string, Preseason=1 Regular Season=2 Postseason=4
    :return: A list of strings containing all gameids from that season
    '''
    game_ids = []
    for i in range(game_number):
        id = '00'
        id += season_part
        id += season
        if i+1 < 10000:
            id += '0'
        if i+1 < 1000:
            id += '0'
        if i+1 < 100:
            id += '0'
        if i+1 < 10:
            id += '0'
        id += str(i+1)
        game_ids.append(id)
    return game_ids

def make_pbp_csv(id):
    '''
    :param ids: game id as str
    :return: saves a csv to computer of dataframe of play by play from given games with given title
    '''
    pbp_df = extract_data(get_pbp_url(id))
    pbp_df.to_csv('data/{}_pbp.csv'.format(id), index=False)

def make_players_on_court_csv(id):
    '''
    :param id: game id as str
    :return: saves a csv to computer of dataframe of starting players at every period of given game
    '''
    players_on_court = get_players_on_court_at_start_of_period_df(id)
    players_on_court.to_csv('data/{}_players_at_period.csv'.format(id), index=False)