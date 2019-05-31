import lightgbm as lgb
import csv
import pandas as pd
import time
import json
# -*- coding: utf-8 -*-
from joblib import load, dump
from train import Train
from tqdm import tqdm
import re
from utils.features_ents import feature_ents
from utils.features_emos import feature_emos
from utils.nerdict import jieba_ner
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, recall_score, precision_score
# from utils.model_lgb import model_lgb
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE 

all_Y_ents = []
test_file = '../data/coreEntityEmotion_train.txt'
test_file = open(test_file)
for news in tqdm(test_file):
    news = json.loads(news)
    Y_ents_emos = news['coreEntityEmotions']
    Y_ents = []
    for y in Y_ents_emos:
        Y_ents.append(y['entity'])
    all_Y_ents.append(Y_ents)
    
def loadData(path):
    res = []
    file = open(path)
    for f in file:
        res.append(json.loads(f))
    return res

count = 0
# ners_appear_times = {}   #自己的分词出现的次数
ners_select_times = {}   #实体被当做真实实体的次数
ners_select_per = {}     #实体被当做真实实体的概率
for Y_ents in all_Y_ents:
    count+=1
    for ent in Y_ents:
        if ners_appear_times[ent] == 0:
            print(ent)
            continue
        if ent in ners_select_times:
            ners_select_times[ent] += 1
            ners_select_per[ent] = ners_select_times[ent]/ners_appear_times[ent]
        else:
            ners_select_times[ent] = 1
            ners_select_per[ent] = 1/ners_appear_times[ent]

dump(ners_select_per,'features/all_ners_select_per.joblib')
dump(ners_select_times,'features/all_ners_select_times.joblib')
print("done")

import time
import scipy.special as special
class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
#         print(sample)
        I = []
        C = []
        for clk_rt in sample:
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):

        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0
#         print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        for i in range(0, len(imps), 1):    # 步长复用时去掉
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))
        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)
def bys(I, C):
    bs = BayesianSmoothing(1, 1)
    # I, C = bs.sample(500, 500, 10, 1000)
    bs.update(I, C, 1000, 0.00001)   #
#     print(bs.alpha, bs.beta)
    ctr = []
    for i in range(len(I)):
        ctr.append((C[i] + bs.alpha) / (I[i] + bs.alpha + bs.beta))
    return ctr

bys_ctr = {}
for ent in tqdm(ners_select_times):
    if ners_select_times[ent] > 9:
        bys_ctr[ent] = bys([ners_select_times[ent]],[ners_select_per[ent]])
    
    
dump(bys_ctr,'features/bys_ctr_10.joblib')   
    
    
    
    
    
    
    
    