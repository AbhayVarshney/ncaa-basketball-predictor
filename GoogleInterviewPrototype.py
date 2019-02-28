from tkinter import *
import pandas as pd
from sklearn import linear_model
import numpy
import math
import random
import csv
import io
import os
from pygame import mixer
from sys import byteorder
from array import array
from struct import pack
import pyaudio
import wave
import time
from PIL import ImageTk, Image

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.cloud import texttospeech

# global
# list of all stats in college basketball game
stats_fields = ['score', 'fga', 'fgp', 'fga3', '3pp', 'ftp', 'or', 'dr', 'ast', 'to', 'stl', 'blk', 'pf']
team_stats = {}
college_basketball_samples = []
binary_win = []
THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100


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


def setUpTourney(seeds, model, total_matchups, team_rating):
    # obtain tournament teams
    tourney_teams = []
    for index, col in seeds.iterrows():
        if col['Season'] == 2017:
            tourney_teams.append(col['Team'])

    # Build our prediction of every matchup.
    print("Analyzing tourney results to begin predicting matchups.")
    tourney_teams.sort()
    for team_1 in tourney_teams:
        for team_2 in tourney_teams:
            if team_2 > team_1:
                label = str(2017) + '_' + str(team_1) + '_' + str(team_2)
                total_matchups.append([label, makePrediction(team_1, team_2, model, 2017, [], team_rating)[0][0]])


def analyzeSeason(season_data, team_rating):
    print("Beginning analyzing Season Data and computing rating based of ELO algorithm.")
    for index, column in season_data:
        isUsable = True
        myYear = column['Season']  # gives year
        if column['Wloc'] == 'H':  # home team gets 100 in ranking
            team_a_ranking = 100 + getRating(column['Wteam'], myYear, team_rating)
            team_b_ranking = getRating(column['Lteam'], myYear, team_rating)
        else: # a loss
            team_a_ranking = getRating(column['Wteam'], myYear, team_rating)
            team_b_ranking = 100 + getRating(column['Lteam'], myYear, team_rating)
        copy_team_a_ranking = [team_a_ranking]
        copy_team_b_ranking = [team_b_ranking]

        for field in stats_fields:
            team_a_stats = calculateStatistics(column['Wteam'], myYear, field)
            team_b_stats = calculateStatistics(column['Lteam'], myYear, field)
            if team_a_stats is 0 and team_b_stats is 0:
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
def handleResponse(input):
    # acquire list of teams
    my_teams = []
    for index, col in pd.read_csv('my_data/Teams.csv').iterrows():
        my_teams.append(col['Team_Name'])

    my_response = [
        'So after some calculation, I believe that %s is going to beat %s, with a likelihood of about %s%%.',
        'I feel like %s is going to win against %s. I think there is about %s%% chance that this happens. Good luck!',
        'I do not know about this but maybe %s is going to beat %s. The chance that this happens is about %s%%. It is your choice!',
        '%s is probably going to beat %s. I think the likelihood of that happening is probably %s%%. Good luck!'
    ]

    matchups = []
    for team in my_teams:
        if team.lower() in input.lower() and len(matchups) < 2:
            matchups.append(team)
    if len(matchups) != 2:
        return "Sorry, the team that you asked about isn't in the NCAA tournament!"
    else:
        csv_file = csv.reader(open('my_data/my_predictions.csv', "rt"), delimiter=",")
        for row in csv_file:
            if matchups[0] == row[0] and matchups[1] == row[1]:
                my_prediction = row[2]
            elif matchups[0] == row[1] and matchups[1] == row[0]:
                my_prediction = row[2]

        if float(my_prediction) < 0.5:
            my_prediction_temp = str(int((1-float(my_prediction)) * 100))
            resp = random.choice(my_response) % (matchups[1], matchups[0], my_prediction_temp)
        else:
            my_prediction_temp = str(int(float(my_prediction) * 100))
            resp = random.choice(my_response) % (matchups[0], matchups[1], my_prediction_temp)
        return resp


def createModel():
    # obtain bball score results from csv (obtained from Kaggle)
    # initialize necessary variables
    print("Initializing Logistic Regression Model for prediction.")
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
    print("Total college basketball samples obtained from csv: [%d]" % len(college_basketball_samples))

    print("Fitting samples to Logistic Regression model.")
    model.fit(college_basketball_samples, binary_win)
    setUpTourney(seeds, model, total_matchups, team_rating)

    # convert data to .csv
    for index, col in pd.read_csv('my_data/Teams.csv').iterrows():
        teamsArr[col['Team_Id']] = col['Team_Name']
    for matchup in total_matchups:
        values = matchup[0].split('_')
        csv_data.append([teamsArr[int(values[1])], teamsArr[int(values[2])], matchup[1]])

    with open('my_data/my_predictions.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)


def googleSpeechToTextQuery():
    output = ""
    file_name = os.path.join('resources', 'userOutputResp.wav')

    # Instantiates a client
    client = speech.SpeechClient()

    # Loads the audio into memory
    with io.open(file_name, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    config = types.RecognitionConfig(
        audio_channel_count=1,
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='en-US')

    # Detects speech in the audio file
    response = client.recognize(config, audio)

    for result in response.results:
        output += result.alternatives[0].transcript
    return output


def clientSpeak(message):
    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.types.SynthesisInput(text=message)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.types.VoiceSelectionParams(
        language_code='en-US',
        ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

    # Select the type of audio file you want returned
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3)

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(synthesis_input, voice, audio_config)

    # The response's audio_content is binary.
    with open('resources/output.mp3', 'wb') as out:
        # Write the response to the output file.
        out.write(response.audio_content)

    # play audio
    mixer.init()
    mixer.music.load("resources/output.mp3")
    mixer.music.play()


def normalize(snd_data):
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r


def trim(snd_data):
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r


def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)
    num_silent = 0
    snd_started = False
    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = max(snd_data) < THRESHOLD

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def recordUser(path):
    print("Starting recording")
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
    print("Completed recording")


class Window(Frame):

    def __init__(self, master = None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    def init_window(self):
        self.master.title("ESPN NCAA Basketball Virtual Assistant")
        self.pack(fill=BOTH, expand=1)
        title = Label(root, text="ESPN NCAA Virtual Assistant: James", font="Times 18")
        title.place(x=50, y=90)
        beginRecordingButton = Button(self, text="Ask Me", command=self.client_exit, width=20)
        beginRecordingButton.place(x=90, y=120)
        load = Image.open("resources/MarchMadness.jpg")
        load = load.resize((160, 70), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=110, y=10)
        createModel()  # Create Linear Regression Model

    def client_exit(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "seventh-magnet-232908-69c8bb828ad7.json"
        os.environ["GCLOUD_PROJECT"] = "seventh-magnet-232908"
        jamesMsg = "Hi there, my name is James. I'm your N C Double A Basketball Predictor assistant. How may I help you?"
        initialSpeech = Label(root, text="James: " + jamesMsg, font="Times 15", wraplength=380, justify=LEFT)
        initialSpeech.place(x=10, y=220)
        clientSpeak(jamesMsg)
        time.sleep(6.3)
        recordUser("resources/userOutputResp.wav")
        print("Analyzing text...")
        userText = googleSpeechToTextQuery()
        userSpeech = Label(root, text="You: " + userText, font="Times 15", wraplength=380, justify=LEFT)
        userSpeech.place(x=10, y=170)
        print("Transcription:", userText)
        resp = handleResponse(userText)
        print("Computer output: ", resp)
        jamesSpeech = Label(root, text="James: " + resp, font="Times 15", wraplength=380, justify=LEFT)
        jamesSpeech.place(x=10, y=220)
        clientSpeak(resp)


root = Tk()
root.geometry("400x300")
app = Window(root)
root.mainloop()
