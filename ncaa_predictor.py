import pandas as pd
from sklearn import cross_validation, linear_model
import numpy
import math
import random
import csv
from twitter import *
import time
import config

# global
# list of all stats in college basketball game
stats_fields = ['score', 'fga', 'fgp', 'fga3', '3pp', 'ftp', 'or', 'dr', 'ast', 'to', 'stl', 'blk', 'pf']
team_stats = {}
college_basketball_samples = []
binary_win = []


# elo ranking algorithm obtained from https://www.geeksforgeeks.org/elo-rating-algorithm/
def getRank(rank):
    if rank > 2400:
        return 16
    elif rank < 2400 and rank >= 2100:
        return 24
    else:
        return 32


def getRating(team, year, team_rating):
    try:
        return team_rating[year][team]
    except:
        try:
            # Get the previous season's ending value.
            team_rating[year][team] = team_rating[year - 1][team]
            return team_rating[year - 1][team]
        except:
            # Get the starter elo.
            team_rating[year][team] = 1600
            return 1600


def makePrediction(team_a, team_b, model, year, features, team_rating):
    team1Rating = getRating(team_a, year, team_rating)
    team2Rating = getRating(team_b, year, team_rating)

    features.append(team1Rating)
    for stat in stats_fields:
        year_stats = calculateStatistics(team_a, year, stat)
        features.append(year_stats)

    features.append(team2Rating)
    for stat in stats_fields:
        year_stats = calculateStatistics(team_b, year, stat)
        features.append(year_stats)

    return model.predict_proba([features])


def calculateStatistics(team, year, stat):
    try:
        feature_sum = sum(team_stats[year][team][stat])
        total_len = float(feature_sum)
        return feature_sum / total_len
    except:
        return 0  # default


def setField(my_type, my_row):
    my_field = {}
    for stat in stats_fields:
        if stat == 'fgp':
            my_field['fgp'] = my_row['Wfgm'] / my_row['Wfga'] * 100
        elif stat == '3pp':
            my_field['3pp'] = my_row['Wfgm3'] / my_row['Wfga3'] * 100
        elif stat == 'ftp':
            my_field['ftp'] = my_row['Wftm'] / my_row['Wfta'] * 100
        else:
            my_field[stat] = my_row[str(my_type + stat)]
    return my_field


def update_stats(season, team, fields):
    if team not in team_stats[season]:
        team_stats[season][team] = {}

    for key, value in fields.items():
        if key not in team_stats[season][team]:
            team_stats[season][team][key] = []

        if len(team_stats[season][team][key]) >= 9:
            team_stats[season][team][key].pop()
        team_stats[season][team][key].append(value)


def combineSamples(team_a, team_b):
    irand = random.randint(0, 9)
    if irand > 4:
        newRank = team_a + team_b
        binary_win.append(0)
    else:
        newRank = team_b + team_a
        binary_win.append(1)
    college_basketball_samples.append(newRank)


def setUpTourney(model):
    # obtain tournament teams
    tourney_teams = []
    for index, col in seeds.iterrows():
        if col['Season'] == 2017:
            tourney_teams.append(col['Team'])

    # Build our prediction of every matchup.
    print("Predicting matchups.")
    tourney_teams.sort()
    for team_1 in tourney_teams:
        for team_2 in tourney_teams:
            if team_2 > team_1:
                # print("%s beats %s. Prediction accuracy: %f." % (team_2, team_1, prediction[0][0]))
                label = str(2017) + '_' + str(team_1) + '_' + str(team_2)
                total_matchups.append([label, makePrediction(team_1, team_2, model, 2017, [], team_rating)[0][0]])


def analyzeSeason(season_data, team_rating):
    print("Analyzing Season Data and computing rating based of ELO algorithm.")
    for index, column in season_data:
        isUsable = True
        myYear = column['Season']  # gives year
        if column['Wloc'] == 'H':  # home team gets 100 in ranking
            team_a_ranking = 100 + getRating(column['Wteam'], myYear, team_rating)
            team_b_ranking = getRating(column['Lteam'], myYear, team_rating)
        else:
            team_a_ranking = getRating(column['Wteam'], myYear, team_rating)
            team_b_ranking = 100 + getRating(column['Lteam'], myYear, team_rating)
        copy_team_a_ranking = [team_a_ranking]
        copy_team_b_ranking = [team_b_ranking]

        for field in stats_fields:
            team_a_stats = calculateStatistics(column['Wteam'], myYear, field)
            team_b_stats = calculateStatistics(column['Lteam'], myYear, field)
            if team_a_stats is  0 and team_b_stats is 0:
                isUsable = False  # can't use these stats
            else:
                copy_team_a_ranking.append(team_a_stats)
                copy_team_b_ranking.append(team_b_stats)

        if isUsable:
            combineSamples(copy_team_a_ranking, copy_team_b_ranking)

        if column['Wfta'] != 0 and column['Lfta'] != 0:
            update_stats(myYear, column['Wteam'], setField('W', column))
            update_stats(myYear, column['Lteam'], setField('L', column))

        winner_rank = getRating(column['Wteam'], myYear, team_rating)
        loser_rank = getRating(column['Lteam'], myYear, team_rating)
        odds = 1 / (1 + math.pow(10, ((winner_rank - loser_rank) * -1) / 400))

        team_rating[myYear][column['Wteam']] = round(winner_rank + (getRank(winner_rank) * (1 - odds)))
        team_rating[myYear][column['Lteam']] = (loser_rank - (round(winner_rank + (getRank(winner_rank) * (1 - odds))) - winner_rank))

    return college_basketball_samples, binary_win


# connect with twitter @NCAA_Predict
def handleTweets():
    username = "NCAA_Predict"
    sleep_time = 1

    auth = OAuth(config.access_key, config.access_secret, config.consumer_key, config.consumer_secret)
    twitter = Twitter(auth=auth)
    stream = TwitterStream(domain="userstream.twitter.com", auth=auth, secure=True)
    tweet_iter = stream.user()

    # acquire list of teams
    my_teams = []
    for index, col in pd.read_csv('my_data/Teams.csv').iterrows():
        my_teams.append(col['Team_Name'])

    my_response = [
        'Hi @%s, so after some calculation, I believe that %s is gonna beat %s, with a likelihood of %s%%. #ncaabasketball',
        'Hey @%s, I feel like %s is gonna win against %s. I think there is a %s%% chance that this happens. Good luck! #ncaabasketball',
        'Yo @%s, I dont know about this but maybe %s is gonna beat %s. The chance that this happens is %s%%. Its your choice! #ncaabasketball',
        'Wassup @%s, %s is probably gonna beat %s. Chance is probably %s%%. Good luck!'
    ]

    for tweet in tweet_iter:
        matchups = []
        if "entities" not in tweet:
            continue

        mentions = tweet["entities"]["user_mentions"]
        mentioned_users = [mention["screen_name"] for mention in mentions]

        if username in mentioned_users:
            text_tweet = tweet['text']
            my_id = tweet['id']
            for team in my_teams:
                if team.lower() in text_tweet.lower() and len(matchups) < 2:
                    matchups.append(team)

            if len(matchups) != 2:
                print("Not enough teams.")
                status = "Sorry, the team that you tweeted at me isn't in NCAA tournament. Please tweet with a correct team!"
            else:
                print ("Successfully obtained reply from: @%s." % tweet["user"]["screen_name"])
                csv_file = csv.reader(open('my_data/my_predictions.csv', "rt"), delimiter=",")
                for row in csv_file:
                    if matchups[0] == row[0] and matchups[1] == row[1]:
                        my_prediction = row[2]
                        print (my_prediction)
                    elif matchups[0] == row[1] and matchups[1] == row[0]:
                        my_prediction = row[2]
                        print (my_prediction)

                # my_prediction_temp = str(float(my_prediction) * 100)
                if float(my_prediction) < 0.5:
                    my_prediction_temp = str((1-float(my_prediction)) * 100)
                    status = random.choice(my_response) % (tweet["user"]["screen_name"], matchups[1], matchups[0], my_prediction_temp)
                else:
                    my_prediction_temp = str(float(my_prediction) * 100)
                    status = random.choice(my_response) % (tweet["user"]["screen_name"], matchups[0], matchups[1], my_prediction_temp)

            print(matchups)
            try:
                twitter.statuses.update(status=status, in_reply_to_status_id=my_id)
            except:
                print ("error")

        time.sleep(sleep_time)


if __name__ == "__main__":

    # obtain bball score results from csv (obtained from Kaggle)
    # initialize necessary variables
    season_data = pd.read_csv('my_data/RegularSeasonDetailedResults.csv')
    tourney_data = pd.read_csv('my_data/TourneyDetailedResults.csv')
    seeds = pd.read_csv('my_data/TourneySeeds.csv')
    frames = [season_data, tourney_data]
    all_data = pd.concat(frames)
    model = linear_model.LogisticRegression()
    team_rating = {}
    total_matchups = []
    teamsArr = {}
    csv_data = []

    # initalize 2d list
    for i in range(1985, 2018):
        team_rating[i] = {}
        team_stats[i] = {}

    # Begin analyzing season and create model
    college_basketball_samples, binary_win = analyzeSeason(all_data.iterrows(), team_rating)

    print("Total samples: %d" % len(college_basketball_samples))

    # Calculate accuracy using cross-validation sklearn
    print("Cross-validation: %f" % cross_validation.cross_val_score(model, numpy.array(college_basketball_samples), numpy.array(binary_win), cv=10, scoring='accuracy', n_jobs=-1).mean())

    print("Fitting samples to Logistic Regression model.")
    model.fit(college_basketball_samples, binary_win)
    setUpTourney(model)

    # convert data to .csv
    print("Converting results to csv.")
    for index, col in pd.read_csv('my_data/Teams.csv').iterrows():
        teamsArr[col['Team_Id']] = col['Team_Name']
    for matchup in total_matchups:
        values = matchup[0].split('_')
        csv_data.append([teamsArr[int(values[1])], teamsArr[int(values[2])], matchup[1]])

    with open('my_data/my_predictions.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print("Beginning Twitter communication...")
    handleTweets()
