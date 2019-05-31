import lightgbm as lgb
import csv
import pandas as pd
import numpy as np
import time
import json
# -*- coding: utf-8 -*-
from joblib import load, dump
from train import Train
from tqdm import tqdm
import re
from sklearn.linear_model import LogisticRegression
from utils.features_ents import feature_ents
from utils.features_emos import feature_emos
from utils.nerdict import jieba_ner
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV   #Perforing grid search
from collections import Counter
from imblearn.over_sampling import SMOTE 
from utils.find_threshold import threshold_search
from utils.easy_find_threshold import easy_threshold_search
import importlib

X_file = pd.read_csv('features/final_v4_ents_x.csv',index_col=0)
Y = load("features/final_v4_ents_y.joblib")
print(len(Y))
print(len(X_file))


x_classes = pd.read_csv('features/final_classes.csv')
np_x_classes = np.array(x_classes)#np.ndarray()
x_classes_list=np_x_classes.tolist()#list

bys = load('features/bys_ctr_10.joblib')
X_file = X_file.drop(['Unnamed: 0.1'],axis=1)

np_X_file = np.array(X_file)#np.ndarray()
train_x_list=np_X_file.tolist()#list

#选特征
select_x = []
select_ent_id = []
tmpid = np.nan
classes = x_classes_list[:40000]
i = -1
c = 0
for x in train_x_list:
    if x[1] in bys:
        byssmooth = bys[x[1]][0]
    else:
        byssmooth = np.nan
    if tmpid != x[0]:
        tmpid = x[0]
        i+=1
        select_x.append(x[2:]+classes[i][1:]+[byssmooth])
        select_ent_id.append(x[0:2])
    else:
        select_x.append(x[2:]+classes[i][1:]+[byssmooth])
        select_ent_id.append(x[0:2])



def model_lgb(X, Y, process_num, evalx, evaly, seed):
    trainx, evalx, trainy, evaly = train_test_split(X, Y, test_size=0.2, random_state=seed)
    X = pd.DataFrame(trainx)
    Y = pd.DataFrame(trainy)
    evalx = pd.DataFrame(evalx)
    evaly = pd.DataFrame(evaly)
    lgb_train = lgb.Dataset(X, Y)
    lgb_eval = lgb.Dataset(evalx, evaly, reference=lgb_train)
    
    params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['xentropy'], #auc
    'num_leaves': 63,
#     'max_depth' : 8,
    'learning_rate': 0.01,
    'feature_fraction': 1,
    'bagging_fraction': 1,
    'bagging_seed': 0,
    'bagging_freq': 1,
    'seed':1024,
    'verbosity':10,
#     'lambda_l1': 0.1,
#     'lambda_l2': 2
    }
    print("Training lgb model....")
    gbm = lgb.train(params, lgb_train,num_boost_round=5000, valid_sets=lgb_eval, early_stopping_rounds=100)
    print("Save model to "+process_num+".joblib")
    dump(gbm, "models/"+process_num+".joblib")

model_lgb(select_x, Y, "final_ents_v1_lgb_1", 0, 0, 1)










