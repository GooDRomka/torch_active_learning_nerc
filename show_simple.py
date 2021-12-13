import pandas as pd
import codecs
import os
from tqdm import tqdm
import numpy as np
import csv
import json
import random
import pylab
from configs import ModelConfig, ActiveConfig
import matplotlib.pyplot as plt

def read_from_csv(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        stats = []
        for line in reader:
            stats.append(line)
        return stats

def read_file_simple(path):
    experiments = []
    loginfo = read_from_csv(path)
    for line in loginfo:
        if len(line) != 0:
            if line[0] == "BEGIN":
                stat = {"budget": float(line[2])
                         }
            if line[0] == "result":
                stat.update({"f1": float(line[16]), "precision": float(line[12]), "recall": float(line[14])})
                experiments.append(stat)
    return pd.DataFrame(experiments)

def find_new_number(directory):
    result = 0
    for filename in os.listdir(directory):
        try:
            num = int(filename[:2])
            result = num if num > result else result
        except Exception:
            pass

    if result+1<10:
        result = "0"+str(result+1)
    else:
        result = str(result+1)
    return result
def random_color():
    levels = range(32,256,32)
    return tuple(random.choice(levels) for _ in range(3))

def read_file_simple_old(path):
    experiments = []
    loginfo = read_from_csv(path)
    for line in loginfo:
        if len(line) != 0:
            if line[0] == "Begin":
                stat = {"budget": float(line[3]), "iter":[]
                         }
            if line[0] == "EndIter":
                stat["iter"].append([line[14],line[6],line[8],line[10],line[16],line[17],line[18]])
            if line[0] == "Results":
                stat.update({"f1": float(line[10]), "precision": float(line[6]), "recall": float(line[8]), "epoch":int(line[14])})
                experiments.append(stat)
    return pd.DataFrame(experiments)


if __name__ == '__main__':
    model_config = ModelConfig()
    directory_report = "report/simple/"
    path_active = 'logs/simple/01_loginfo.csv'
    if not os.path.exists(directory_report):
        os.makedirs(directory_report)
    new_plot_num = find_new_number(directory_report)

    path_simple = "logs/simple/01_loginfo.csv"
    mode = 'simple'

    experiments = read_file_simple(path_simple)
    print(experiments)
    experiments_simple = experiments.groupby('budget', as_index=False).agg({'f1': ['mean', 'std'],'precision': ['mean', 'std'],'recall': ['mean', 'std']})

    print(experiments_simple)

    plt.figure(figsize=(10,9))
    plt.plot(experiments_simple['budget'], experiments_simple[('f1','mean')], marker="o", label="TORCH")
    plt.fill_between(experiments_simple['budget'],experiments_simple[('f1','mean')]+experiments_simple[('f1','std')],experiments_simple[('f1','mean')]- experiments_simple[('f1','std')],alpha=.2)

    #
    # experiments_old = read_file_simple_old("logs/simple/loginfo_simple_batch_8.csv")
    # experiments_simple = experiments_old.groupby('budget',as_index=False).agg({'f1': ['mean', 'std'],'precision': ['mean', 'std'],'recall': ['mean', 'std'],"epoch":['mean','std']})
    #
    #
    # plt.plot(experiments_simple['budget'], experiments_simple[('f1','mean')], marker="o", label="TF")
    # plt.fill_between(experiments_simple['budget'],experiments_simple[('f1','mean')]+experiments_simple[('f1','std')],experiments_simple[('f1','mean')]- experiments_simple[('f1','std')],alpha=.2)
    # plt.legend(loc='best')
    # plt.xlabel('budget')
    # plt.ylabel('f1')
    # plt.savefig(directory_report+new_plot_num+'simple.png')





