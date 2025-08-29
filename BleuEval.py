from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor_score as ms
import nltk
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import random

def calcScores(sysName):
    weights = (0.25, 0.25, 0.25, 0.25)

    translatedText = open(f"experiment/europarl.{sysName}.results.en", "r")
    referenceText = open("experiment/europarl.experiment.en", "r")

    translatedLines = []
    referenceLines = []
    meteorScores = []
    bleuScores = []

    count = 1
    for tLine, rLine in zip(translatedText, referenceText):
        tToken = tLine.strip().split()
        rToken = rLine.strip().split()
        translatedLines.append(tToken)
        referenceLines.append([rToken])
        s = ms.single_meteor_score(rToken, tToken)
        meteorScores.append(round(s, 5))
        b = sentence_bleu([rToken], tToken, weights=weights)
        bleuScores.append(round(b, 5))
        count += 1
    return bleuScores, meteorScores

def bootstrap(sysName, meteorScores: list, bleuScores: list):
    meteorMeans = []
    bleuMeans = []
    for i in range(1000):
        mBootstrap = random.choices(meteorScores, k=60000)
        bBootstrap = random.choices(bleuScores, k=60000)
        meteorMeans.append(np.mean(mBootstrap))
        bleuMeans.append(np.mean(bBootstrap))
    
    return bleuMeans, meteorMeans

if __name__ == "__main__":
    systemsList = ["llama", "m2m100", "marian", "nllb", "t5", "apertium"]
    bootstrapAverages = {}
    realAverage = {}
    title = ""
    for system in systemsList:
        bleu, meteor = calcScores(system)
        bleuMeans, meteorMeans = bootstrap(system, meteor, bleu)

        #Unique average of all individual sentences for a system
        realAverage[system] = [np.mean(bleu), np.mean(meteor)]

        #List of 1000 bootstrap averages
        bootstrapAverages[system] = [bleuMeans, meteorMeans]

        #Creating title line of csv file
        title += f"{system}_Bleu,{system}_Meteor,"
    
    title = title[:-1]

    #Loop to write data to the csv
    with open("translation-bootstrap.csv", "w") as file:
        file.write(title + "\n")
        for i in range(1000):
            newLine = ""
            for sys in systemsList:
                newLine += str(bootstrapAverages[sys][0][i]) + "," + str(bootstrapAverages[sys][1][i]) + ","
            file.write(newLine[:-1] + "\n")
    
    file.close()
    averageRes = open("total-averages.csv", "w")
    averageRes.write(title + "\n")
    data = ""
    for sys in systemsList:
        data += str(realAverage[sys][0]) + "," + str(realAverage[sys][1]) + ","
    averageRes.write(data[:-1])