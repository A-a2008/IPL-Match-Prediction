from django.shortcuts import render
from joblib import load
from .ai.ai import transform_dataset, give_full_name, give_short_names
import random

# Create your views here.


def home(request):
    return render(request, "index.html")


def predict_winner(venue, team1, team2, toss_winner, toss_decision):
    dataset = {
        "venue": [venue],
        "team1": [team1],
        "team2": [team2],
        "toss_winner": [toss_winner],
        "toss_decision": [toss_decision],
        "method": [float("NaN")]
    }
    nbrs = load("main/models/KNeighborsClassifierModel.joblib")
    lgr = load("main/models/LogisticRegressionModel.joblib")
    sgdc = load("main/models/SGDClassifierModel.joblib")
    predicting_dataset = transform_dataset(dataset)
    nbrs_pred = nbrs.predict(predicting_dataset)
    lgr_pred = lgr.predict(predicting_dataset)
    sgdc_pred = sgdc.predict(predicting_dataset)
    if nbrs_pred == lgr_pred:
        return give_full_name(nbrs_pred[0])
    elif lgr_pred == sgdc_pred:
        return give_full_name(lgr_pred[0])
    elif nbrs_pred == sgdc_pred:
        return give_full_name(nbrs_pred[0])
    else:
        guessing_list = [give_full_name(nbrs_pred[0]), give_full_name(lgr_pred[0]), give_full_name(sgdc_pred[0])]
        return f"Could not determine the winning team \nBut I can give a guess, and that is {random.choice(guessing_list)}"


def predict(request):
    venue = request.POST['venue']
    team1 = request.POST['team1']
    team2 = request.POST['team2']
    toss_winner = request.POST['toss_winner']
    toss_decision = request.POST['toss_decision']
    if toss_decision.lower() == "bowl":
        toss_decision = "field"

    try:
        winner = predict_winner(venue, team1, team2, toss_winner, toss_decision)
        data = {
            'winner': winner,
            'short_name': give_short_names(winner)
        }

        return render(request, "index.html", data)
    except ValueError:
        data = {
            "error_message": "Value error"
        }

        return render(request, "index.html", data)


def how_to_use(request):
    return render(request, "how-to-use-ai.html")



