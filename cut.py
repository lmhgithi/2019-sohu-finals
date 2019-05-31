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
from utils.features_ents import feature_ents
from utils.features_emos import feature_emos
from utils.nerdict import jieba_ner
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV   #Perforing grid search
from collections import Counter
from imblearn.over_sampling import SMOTE 

def loadData(path):
    res = []
    file = open(path)
    for f in file:
        res.append(json.loads(f))
    return res



# 这块是队友tianjialai的bert结果
with open("data/bert10train.txt","r",encoding="utf-8") as w:
    txt=w.readlines()
txt=[i.strip() for i in txt]
alre=[i.replace("##","").split(",") for i in txt]
bert=[]
for i in alre:
    while "" in i:
        i.remove("")
    bert.append(i)
    
with open("data/bert10fold.txt","r",encoding="utf-8") as w:
    txt=w.readlines()
txt=[i.strip() for i in txt]
alre=[i.replace("##","").split(",") for i in txt]
bert2=[]
for i in alre:
    while "" in i:
        i.remove("")
    bert2.append(i)
    
stopwordfile = open("./data/stopwords.txt")
stopwords = set()
for stp in stopwordfile:
    stopwords.add(re.sub('\n','',stp))
    
    
    
#结巴分词
fea_ents = feature_ents(-2)
myjieba = fea_ents.getjieba()
test_file = '../data/coreEntityEmotion_train.txt'
has_num = 0
tal_num = 0
count = 0
ttc = 0
ners_totxt = []
for news in tqdm(loadData(test_file)):
    count += 1
#     if count == 40000:
#         break
    if count >= 0:
        title = news['title']
        content = news['content']
        ents = []
        ner_list = myjieba.cut(title+','+content)
        for ner in ner_list:
            if len(ner) > 1 and not re.sub('[.]','',ner).isdigit():
                ents.append(ner)

        #标题找英文
        tmp = re.findall("[a-zA-Z0-9]+[ ]?[a-zA-Z0-9]+",title+','+content)
        for t in tmp:
            if not re.sub('[.]','',t).isdigit():
                ents.append(t)
#         ents = list(set(ents))
        for i in range(len(ents[:250])):
            for j in range(len(ents[:250])):
                if i!=j:
                    if ents[i] not in stopwords and ents[j] not in stopwords:
                        if content.count(ents[i]+ents[j])>1:
                            ents.append(ents[i]+ents[j])
                        if content.count(ents[i]+' '+ents[j])>1:
                            ents.append(ents[i]+' '+ents[j])
#                             print(ents[i],ents[j],'->',ents[i]+ents[j])
        ents= sorted(set(ents),key=ents.index)
#         ents = list(set(ents))
        select_ents = []
        for ent in ents:
#             if len(ent) <= 1:
#                 continue
            if re.sub('[.]','',ent).isdigit():
                continue
            if re.sub(r'[0-9一二三四五六七八九十]+[年月日亿万千百十个元%]+', '', ent) == '':
                continue
            select_ents.append(ent)
            ttc += 1
        ners_totxt.append(select_ents)

        Y_ents_emos = news['coreEntityEmotions']
        Y_ents = []
        for y in Y_ents_emos:
            Y_ents.append(y['entity'])
        for y in Y_ents:
            tal_num += 1
            if y in select_ents:
                has_num += 1
        if count %1000 == 0:
             print("tmp覆盖率：",count,has_num/tal_num)
print("覆盖率：",has_num/tal_num)

dump(ners_totxt,'data/final_cut_train_v2.joblib')


fea_ents = feature_ents(1)
myjieba = fea_ents.getjieba()
file = '../data/coreEntityEmotion_test_stage2.txt'
count = 0
ttc = 0
ners_totxt = []
for news in tqdm(loadData(file)):
    count += 1
    if count >= 0:
        title = news['title']
        content = news['content']
        ents = []
        ner_list = myjieba.cut(title+','+content)
        for ner in ner_list:
            if len(ner) > 1 and not re.sub('[.]','',ner).isdigit():
                ents.append(ner)
        #标题找英文
        tmp = re.findall("[a-zA-Z0-9]+[ ]?[a-zA-Z0-9]+",title+','+content)
        for t in tmp:
            if not re.sub('[.]','',t).isdigit():
                ents.append(t)
        for i in range(len(ents[:250])):
            for j in range(len(ents[:250])):
                if i!=j:
                    if ents[i] not in stopwords and ents[j]:
                        if content.count(ents[i]+ents[j])>1:
                            ents.append(ents[i]+ents[j])
        ents= sorted(set(ents),key=ents.index)
        select_ents = []
        for ent in ents:
            if re.sub('[.]','',ent).isdigit():
                continue
            if re.sub(r'[0-9一二三四五六七八九十]+[年月日亿万千百十个元]+', '', ent) == '':
                continue
            select_ents.append(ent)
            ttc += 1
        ners_totxt.append(select_ents)
dump(ners_totxt,'data/final_cut_test_v2.joblib')




#加上bert的词
ners_totxt = []
for n,b in zip(ners,bert):
    if count >= 0:
        ents = []
        ner_list = n
        for ner in ner_list:
            if len(ner) > 1 and not re.sub('[.]','',ner).isdigit():
                ents.append(ner)
        b = sorted(set(b),key=b.index)
        for ner in b:
            ents.append(ner)
        ents= sorted(set(ents),key=ents.index)
#         ents = list(set(ents))
        select_ents = []
        for ent in ents:
            if re.sub('[.]','',ent).isdigit():
                continue
            if re.sub(r'[0-9一二三四五六七八九十]+[年月日亿万千百十个元%]+', '', ent) == '':
                continue
            select_ents.append(ent)
            ttc += 1
        ners_totxt.append(select_ents)

dump(ners_totxt,'data/final_train_cut_v3.joblib')

ners_totxt = []
for n,b in zip(test_ners,bert2):
    if count >= 0:
        ents = []
        ner_list = n
        for ner in ner_list:
            if len(ner) > 1 and not re.sub('[.]','',ner).isdigit():
                ents.append(ner)
        b = sorted(set(b),key=b.index)
        for ner in b:
            ents.append(ner)
        ents= sorted(set(ents),key=ents.index)
#         ents = list(set(ents))
        select_ents = []
        for ent in ents:
            if re.sub('[.]','',ent).isdigit():
                continue
            if re.sub(r'[0-9一二三四五六七八九十]+[年月日亿万千百十个元%]+', '', ent) == '':
                continue
            select_ents.append(ent)
            ttc += 1
        ners_totxt.append(select_ents)
        
dump(ners_totxt,'data/final_test_cut_v3.joblib')