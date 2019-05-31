import lightgbm as lgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
import importlib
import utils
import multiprocessing
#这些是分词，来自于cut.py文件
train_ners = load('data/final_train_cut_v3.joblib')
test_ners = load('data/final_test_cut_v3.joblib')

def loadData(path):
    res = []
    file = open(path)
    for f in file:
        res.append(json.loads(f))
    return res

#训练集特征
def worker(startnum, endnum, processnum):
    print("startnum->",startnum,"endnum->",endnum,"processnum->",processnum)
    train_data_path = "data/coreEntityEmotion_train.txt"
    global train_ners
    fea_ents = feature_ents(-2)
    result = pd.DataFrame()
    count = 0
    for news, ner in zip(loadData(train_data_path), train_ners):
        count += 1
        if count == endnum:
            break
        if count >= startnum :
            st = time.time()
            ent_fea1 = fea_ents.combine_features(news, ner)
            ent_fea2 = fea_ents.combine_new_features(news)
            combfea = pd.concat([ent_fea1, ent_fea2], axis=1, sort=False,join='inner')
            result = pd.concat([result, combfea], axis=0, ignore_index=True, sort=False)
            et = time.time()
            print(count,'/',str(processnum),' ',(processnum-count)/2500,"train",et-st)
    result.to_csv("features/final_v4_ents_x_"+str(processnum)+".csv",header=True)
    print("done train"+str(processnum)+" set")


process = []
count = 0
for i in range(0,40001):
    if i > 0 and i % 5000 == 0:
        p = multiprocessing.Process(target=worker, args=(i-5000+1, i+1, i, ))
        process.append(p)
        p.start()
print("done starting workers")




#测试集特征
def worker(startnum, endnum, processnum):
    print("startnum->",startnum,"endnum->",endnum,"processnum->",processnum)
    test_data_path = 'data/coreEntityEmotion_test_stage2.txt'
    global test_ners
    fea_ents = feature_ents(1)
    result = pd.DataFrame()
    count = 0
    et = time.time()
    for news, ner in zip(loadData(test_data_path), test_ners):
        count += 1
        if count == endnum :
            break
        if count >= startnum :
            st = time.time()
            ent_fea1 = fea_ents.combine_features(news, ner)
            ent_fea2 = fea_ents.combine_new_features(news)
            combfea = pd.concat([ent_fea1, ent_fea2], axis=1, sort=False,join='inner')
            result = pd.concat([result, combfea], axis=0, ignore_index=True, sort=False)
            et = time.time()
            print(count,'/',str(processnum),' ',(processnum-count)/5000,"test",et-st)
    result.to_csv("features/final_v4_test_ents_x_"+str(processnum)+".csv",header=True)
    print("done test"+str(processnum)+" set")


process2 = []
count = 0
for i in range(0,80001):
    if i > 0 and i % 5000 == 0:
        p = multiprocessing.Process(target=worker, args=(i-5000+1, i+1, i, ))
        process.append(p)
        p.start()
print("done starting workers")


#整合多进程文件
X_file2 = pd.read_csv('features/final_v4_ents_x_5000.csv')
X_file4 = pd.read_csv('features/final_v4_ents_x_10000.csv')
X_file6 = pd.read_csv('features/final_v4_ents_x_15000.csv')
X_file8 = pd.read_csv('features/final_v4_ents_x_20000.csv')
X_file10 = pd.read_csv('features/final_v4_ents_x_25000.csv')
X_file12 = pd.read_csv('features/final_v4_ents_x_30000.csv')
X_file14 = pd.read_csv('features/final_v4_ents_x_35000.csv')
X_file16 = pd.read_csv('features/final_v4_ents_x_40000.csv')
print(1)
X_file = pd.DataFrame()
X_file = pd.concat([X_file, X_file2], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file4], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file6], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file8], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file10], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file12], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file14], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file16], axis=0, ignore_index=True, sort=False)
print(len(X_file))
X_file.to_csv('features/final_v4_ents_x.csv',header=True)
print(3)


X_file1 = pd.read_csv('features/final_v4_test_ents_x_5000.csv')
X_file2 = pd.read_csv('features/final_v4_test_ents_x_10000.csv')
X_file3 = pd.read_csv('features/final_v4_test_ents_x_15000.csv')
X_file4 = pd.read_csv('features/final_v4_test_ents_x_20000.csv')
X_file5 = pd.read_csv('features/final_v4_test_ents_x_25000.csv')
X_file6 = pd.read_csv('features/final_v4_test_ents_x_30000.csv')
X_file7 = pd.read_csv('features/final_v4_test_ents_x_35000.csv')
X_file8 = pd.read_csv('features/final_v4_test_ents_x_40000.csv')
X_file9 = pd.read_csv('features/final_v4_test_ents_x_45000.csv')
X_file10 = pd.read_csv('features/final_v4_test_ents_x_50000.csv')
X_file11 = pd.read_csv('features/final_v4_test_ents_x_55000.csv')
X_file12 = pd.read_csv('features/final_v4_test_ents_x_60000.csv')
X_file13 = pd.read_csv('features/final_v4_test_ents_x_65000.csv')
X_file14 = pd.read_csv('features/final_v4_test_ents_x_70000.csv')
X_file15 = pd.read_csv('features/final_v4_test_ents_x_75000.csv')
X_file16 = pd.read_csv('features/final_v4_test_ents_x_80000.csv')
print(1)
X_file = pd.DataFrame()
X_file = pd.concat([X_file, X_file1], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file2], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file3], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file4], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file5], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file6], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file7], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file8], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file9], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file10], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file11], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file12], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file13], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file14], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file15], axis=0, ignore_index=True, sort=False)
X_file = pd.concat([X_file, X_file16], axis=0, ignore_index=True, sort=False)
print(len(X_file))
X_file.to_csv('features/final_v4_test_ents_x.csv',header=True)
print(3)


















