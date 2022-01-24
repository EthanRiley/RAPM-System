import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# a list of lambdas for cross validation
lambdas_rapm = [.01, .05, .1]


def build_player_list(posessions):
    '''
    :param posessions: Possession data frame
    :return: Players present in pbp data
    Function from Ryan Davis
    '''
    players = list(
        set(list(posessions['offensePlayer1Id'].unique()) + list(posessions['offensePlayer2Id'].unique()) + list(
            posessions['offensePlayer3Id']) + \
            list(posessions['offensePlayer4Id'].unique()) + list(posessions['offensePlayer5Id'].unique()) + list(
            posessions['defensePlayer1Id'].unique()) + \
            list(posessions['defensePlayer2Id'].unique()) + list(posessions['defensePlayer3Id'].unique()) + list(
            posessions['defensePlayer4Id'].unique()) + \
            list(posessions['defensePlayer5Id'].unique())))
    players.sort()
    return players

def adjust_to_per_poss(possessions, column):
    '''
    :param possessions: Possession data frame
    :param column: string, The column header to be adjusted to per 100
    :return: Dictionary with per possession appended
    '''
    cpp = column
    cpp += ' per possession'
    possessions[cpp] = 100 * possessions[column] / possessions[possessions]
    return possessions

# Will need to convert player ids into dummy variable row for the training matrix
def map_players(row_in, players):
    '''
    :param row_in: player id row
    :param players: player list
    :return: matrix containing appropriate dummy variables for each player id on each possession
    '''
    p1 = row_in[0]
    p2 = row_in[1]
    p3 = row_in[2]
    p4 = row_in[3]
    p5 = row_in[4]
    p6 = row_in[5]
    p7 = row_in[6]
    p8 = row_in[7]
    p9 = row_in[8]
    p10 = row_in[9]

    rowOut = np.zeros([len(players) * 2])

    rowOut[players.index(p1)] = 1
    rowOut[players.index(p2)] = 1
    rowOut[players.index(p3)] = 1
    rowOut[players.index(p4)] = 1
    rowOut[players.index(p5)] = 1

    rowOut[players.index(p6) + len(players)] = -1
    rowOut[players.index(p7) + len(players)] = -1
    rowOut[players.index(p8) + len(players)] = -1
    rowOut[players.index(p9) + len(players)] = -1
    rowOut[players.index(p10) + len(players)] = -1

    return rowOut

def generate_pbp_matrix(possessions, name, players):
    '''
    :param possessions: Parsed possessions file
    :param name: name
    :param players: players list
    :return: possession matrix for RAPM calculation
    '''
    # Player IDs into matrix
    x_base = possessions.as_matrix(columns=['offensePlayer1Id', 'offensePlayer2Id',
                                                   'offensePlayer3Id', 'offensePlayer4Id', 'offensePlayer5Id',
                                                   'defensePlayer1Id', 'defensePlayer2Id', 'defensePlayer3Id',
                                                   'defensePlayer4Id', 'defensePlayer5Id'])
    # Map matrix to base using Ryan's mapping function
    x_rows = np.apply_along_axis(map_players, 1, x_base, players)
    # Target values into numpy_matrix
    y_rows = possessions.as_matrix([name])
    # List of possessions
    poss_vector = possessions[possessions]
    return x_rows, y_rows, poss_vector

def lambda_to_alpha(lambda_value, samples):
    '''
    turns lambda into alpha value for ridge CV
    '''
    return (lambda_value * samples) / 2.0


# Convert RidgeCV alpha back into a lambda value
def alpha_to_lambda(alpha_value, samples):
    return (alpha_value * 2.0) / samples



def calculate_rapm(train_x, train_y, possessions, lambdas, name, players):
    '''
    :param train_x: nxm training matrix
    :param train_y: nxm training matrixk
    :param possessions: nx1 target matrix
    :param lambdas: list of lambdas
    :param name: name we want to give the value
    :param players: list of players
    :return: RAPM value
    '''
    # convert our lambdas to alphas
    alphas = [lambda_to_alpha(l, train_x.shape[0]) for l in lambdas]

    # create a 5 fold CV ridgeCV model. Our target data is not centered at 0, so we want to fit to an intercept.
    clf = RidgeCV(alphas=alphas, cv=5, fit_intercept=True, normalize=False)

    # fit our training data
    model = clf.fit(train_x, train_y, sample_weight=possessions)

    # convert our list of players into a mx1 matrix
    player_arr = np.transpose(np.array(players).reshape(1, len(players)))

    # extract our coefficients into the offensive and defensive parts
    coef_offensive_array = np.transpose(model.coef_[:, 0:len(players)])
    coef_defensive_array = np.transpose(model.coef_[:, len(players):])

    # concatenate the offensive and defensive values with the playey ids into a mx3 matrix
    player_id_with_coef = np.concatenate([player_arr, coef_offensive_array, coef_defensive_array], axis=1)
    # build a dataframe from our matrix
    players_coef = pd.DataFrame(player_id_with_coef)
    intercept = model.intercept_

    # apply new column names
    players_coef.columns = ['playerId', '{0}__Off'.format(name), '{0}__Def'.format(name)]

    # Add the offesnive and defensive components together (we should really be weighing this to the number of offensive
    # and defensive possession played as they are often not equal).
    players_coef[name] = players_coef['{0}__Off'.format(name)] + players_coef['{0}__Def'.format(name)]

    # rank the values
    players_coef['{0}_Rank'.format(name)] = players_coef[name].rank(ascending=False)
    players_coef['{0}__Off_Rank'.format(name)] = players_coef['{0}__Off'.format(name)].rank(ascending=False)
    players_coef['{0}__Def_Rank'.format(name)] = players_coef['{0}__Def'.format(name)].rank(ascending=False)

    # add the intercept for reference
    players_coef['{0}__intercept'.format(name)] = intercept[0]

    return players_coef, intercept

# Here are some prefiltered possessions for RAPM from Ryan Davis, I wasn't able to get the parser working in time for
# the presentation so I used this data to make the RAPM data I showed in the presentation
possessions = pd.read_csv('data/rapm_possessions.csv')
# build_player_list(possessions).to_csv('data/player_names.csv', index=False)
players = pd.read_csv('data/player_names.csv')

# The data I downloaded was parsed differently: some possessions are 0 possession possessions where nothing happens
# I will just filter out the possessions that aren't actually possessions

possessions = possessions[possessions['possessions'] > 0]

possessions = adjust_to_per_poss(possessions, 'points')

# extract the training data from our possession data frame
train_x, train_y, possessions_raw = generate_pbp_matrix(possessions, 'points per possession', players)

# calculate the RAPM
results, intercept = calculate_rapm(train_x, train_y, possessions_raw, lambdas_rapm, 'RAPM', players)

# round to 2 decimal places for display
results = np.round(results, decimals=2)

# sort the columns
results = results.reindex(sorted(results.columns), axis=1)

# join back with player names
results = players.merge(results, how='inner', on='playerId')

# save as CSV
# results.to_csv('data/rapm.csv')
