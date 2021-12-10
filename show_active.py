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

def read_file_active(path, scale):
    experiments = []
    loginfo = read_from_csv(path)
    for line in loginfo:
        if len(line) > 1:
            if line[0] == "BEGIN":
                stat = {"strategy": line[2], "label_strategy": line[4],"budget":line[6],"init_budget":line[8],"step_budget":line[10],
                        "threshold": line[12], "seed": line[14], "active_iteration": [], "epoch_iter": []}
                budget = float(line[6])
                init_budget = float(line[8])
                step_budget= float(line[10])
                spent_budget = init_budget
                fullcost = init_budget

            if line[0] == "TrainInitFinished":
                stat['active_iteration'].append({"bestf1dev": float(line[12]), "bestprecisiondev": float(line[8]), "bestrecalldev": float(line[10]),"budget": budget, "init_budget": init_budget, "step_budget":step_budget ,"spent_budget":spent_budget})

            if line[0] == "SelectIterFinished":
                spentprice = float(line[15])
                spent_budget = spent_budget+step_budget*scale
                stat['active_iteration'].append({"bestf1dev": float(line[21]), "bestprecisiondev": float(line[17]), "bestrecalldev": float(line[19]),"budget": budget, "init_budget": init_budget, "step_budget":step_budget ,"spent_budget":spent_budget})
                fullcost = float(line[5])

            if line[0] == "result":
                stat.update({"f1test": float(line[12]), "precisiontest": float(line[8]), "recalltest": float(line[10]), "cost_of_train": fullcost,
                             "devf1": float(line[18]), "devprecision": float(line[14]), "devrecall": float(line[16]), "spentprice": spentprice})
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

if __name__ == '__main__':
    model_config = ModelConfig()
    directory_report = "report/active/"
    path_active = 'logs/active/01_loginfo.csv'
    if not os.path.exists(directory_report):
        os.makedirs(directory_report)
    new_plot_num = find_new_number(directory_report)

    path_simple = "logs/active/01_loginfo.csv"


    i = 0
    scales = [0.2, 0.5, 1]
    pli = 1
    plt.style.use('ggplot')


    ####DRAWING SEPARATELY and in a file
    for scale in scales:
        experiments = read_file_active(path_active, scale)
        iterations_c = experiments["active_iteration"]
        iterations = []
        for lis in iterations_c:
            for it in lis:
                iterations.append(it)
        iterations = pd.DataFrame(iterations)

        iterations = iterations.groupby(['budget', 'init_budget','step_budget', 'spent_budget'],as_index=False).agg({'bestf1dev': ['mean','std'], 'bestprecisiondev': ['mean','std'], 'bestrecalldev': ['mean','std']})
        experiments = experiments.groupby(['budget','init_budget','step_budget']).agg(
            {'devf1': ['mean', 'std'], 'devprecision': ['mean', 'std'], 'devrecall': ['mean', 'std']})
        init_budget, budget, step_budget = pd.unique(iterations['init_budget']),pd.unique(iterations['budget']),pd.unique(iterations['step_budget'])

        plt.figure(figsize=(22,16))
        filt = max(budget)*scale+max(init_budget)+1000

        # experiments_simple_filt = experiments_simple[experiments_simple['budget']<=filt]
        # plt.plot(experiments_simple_filt['budget'], experiments_simple_filt[('f1','mean')],label="simple", marker="o")
        # plt.fill_between(experiments_simple_filt['budget'],experiments_simple_filt[('f1','mean')]+experiments_simple_filt[('f1','std')],experiments_simple_filt[('f1','mean')]- experiments_simple_filt[('f1','std')],alpha=.2)

        for i in init_budget:
            df = iterations[iterations['init_budget']==i]
            # print(df)
            plt.plot(df['spent_budget'],df[('bestf1dev','mean')], label=str(i), marker="o")
            plt.fill_between(df['spent_budget'],df[('bestf1dev','mean')]+df[('bestf1dev','std')],df[('bestf1dev','mean')]- df[('bestf1dev','std')],alpha=.2)
            # plt.errorbar(df['spent_budget'],df[('bestf1dev','mean')], df[('bestf1dev','std')], linestyle='None', marker='^')
        plt.xlabel('spent_budget')
        plt.ylabel('bestf1dev')
        plt.legend(loc='best')
        plt.title("active learning with scale = "+str(scale))
        # plt.show()
        plt.savefig(directory_report+new_plot_num+"_active "+ str(scale)+'.png')


    ####DRAWING on a one picture

    for scale in scales:
        pylab.subplot(3, 2, pli)
        experiments = read_file_active(path_active,scale)
        iterations_c = experiments["active_iteration"]
        iterations = []
        for lis in iterations_c:
            for it in lis:
                iterations.append(it)
        iterations = pd.DataFrame(iterations)
        iterations = iterations.groupby(['budget', 'init_budget','step_budget', 'spent_budget'],as_index=False).agg({'bestf1dev': ['mean','std'], 'bestprecisiondev': ['mean','std'], 'bestrecalldev': ['mean','std']})
        experiments = experiments.groupby(['budget','init_budget','step_budget']).agg(
            {'devf1': ['mean', 'std'], 'devprecision': ['mean', 'std'], 'devrecall': ['mean', 'std']})
        init_budget, budget, step_budget = pd.unique(iterations['init_budget']),pd.unique(iterations['budget']),pd.unique(iterations['step_budget'])

        filt = max(budget)*scale+max(init_budget)+1000

        # experiments_simple_filt = experiments_simple[experiments_simple['budget']<=filt]
        # plt.plot(experiments_simple_filt['budget'], experiments_simple_filt[('f1','mean')],label="simple", marker="o")
        # plt.fill_between(experiments_simple_filt['budget'],experiments_simple_filt[('f1','mean')]+experiments_simple_filt[('f1','std')],experiments_simple_filt[('f1','mean')]- experiments_simple_filt[('f1','std')],alpha=.2)

        for i in init_budget:
            df = iterations[iterations['init_budget']==i]
            # print(df)
            pylab.plot(df['spent_budget'],df[('bestf1dev','mean')], label=str(i), marker="o")
            pylab.fill_between(df['spent_budget'],df[('bestf1dev','mean')]+df[('bestf1dev','std')],df[('bestf1dev','mean')]- df[('bestf1dev','std')],alpha=.2)
            # plt.errorbar(df['spent_budget'],df[('bestf1dev','mean')], df[('bestf1dev','std')], linestyle='None', marker='^')
        pylab.xlabel('spent_budget')
        pylab.ylabel('bestf1dev')
        pylab.legend(loc='best')
        pylab.title("active learning with scale = "+str(scale))
        # plt.show()
        # pylab.savefig("report/active "+ str(scale)+'.png')
        pli+=1
    pylab.savefig(directory_report+new_plot_num+"_active_all.png")


    print(experiments)





