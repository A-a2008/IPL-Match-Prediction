import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import load
import random

df = pd.read_csv("main/ai/IPL Matches 2008-2020.csv")
data = df[:]

data = data.replace("Bangalore", "Bengaluru")
data = data.replace("M Chinnaswamy Stadium", "M.Chinnaswamy Stadium")
data = data.replace("D/L", 1)
data = data.replace("Delhi Daredevils", "Delhi Capitals")
data = data.replace("Deccan Chargers", "Sunrisers Hyderabad")
data = data.replace("Royal Challengers Bangalore", "RCB")
data = data.replace("Delhi Capitals", "DC")
data = data.replace("Sunrisers Hyderabad", "SRH")
data = data.replace("Chennai Super Kings", "CSK")
data = data.replace("Kolkata Knight Riders", "KKR")
data = data.replace("Mumbai Indians", "MI")
data = data.replace("Kings XI Punjab", "PBKS")
data = data.replace("Rajasthan Royals", "RR")
columns_to_drop = ['id', 'city', 'date', 'player_of_match', 'neutral_venue', 'result', 'result_margin', 'umpire1', 'umpire2', 'eliminator']
for column in columns_to_drop:
    data = data.drop(column, axis=1)

data['venue'].fillna("missing", inplace=True)
data['team1'].fillna("missing", inplace=True)
data['team2'].fillna("missing", inplace=True)
data['toss_winner'].fillna("missing", inplace=True)
data['toss_decision'].fillna("missing", inplace=True)
data['winner'].fillna("missing", inplace=True)

X = data.drop("winner", axis=1)
y = data['winner']

categorical_features = ['venue', 'team1', 'team2', 'toss_decision', 'toss_winner', 'method']
one_hot = OneHotEncoder()

transformer = ColumnTransformer([('one_hot',
                                 one_hot,
                                 categorical_features)],
                                 remainder='passthrough')
transformer.fit(X)


def give_full_name(team_short_name):
    team_names = {
        "RCB": "Royal Challengers Bangalore",
        "DC": "Delhi Capitals",
        "SRH": "Sunrisers Hyderabad",
        "CSK": "Chennai Super Kings",
        "KKR": "Kolkata Knight Riders",
        "MI": "Mumbai Indians",
        "PBKS": "Punjab Kings",
        "RR": "Rajasthan Royals",
    }

    return team_names[team_short_name]


def give_short_names(team_long_name):
    team_names = {
        "Royal Challengers Bangalore": "RCB",
        "Delhi Capitals": "DC",
        "Sunrisers Hyderabad": "SRH",
        "Chennai Super Kings": "CSK",
        "Kolkata Knight Riders": "KKR",
        "Mumbai Indians": "MI",
        "Punjab Kings": "PBKS",
        "Rajasthan Royals": "RR"
    }

    return team_names[team_long_name]


def transform_dataset(sample_dataset_dict):
    df_to_transform = pd.DataFrame(sample_dataset_dict)
    transformed_df = transformer.transform(df_to_transform).toarray()
    return transformed_df
