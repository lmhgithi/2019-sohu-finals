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

X_file = pd.read_csv('features/final_v4_test_ents_x.csv',index_col=0)
len(X_file)

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
classes = x_classes_list[40000:]
i = -1
c = 0
for x in tqdm(train_x_list):
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
len(select_x)

st = time.time()
ents_model = load('models/final_ents_v1_lgb_1.joblib')
nt_score = ents_model.predict(select_x, num_iteration=ents_model.best_iteration)
print('done1')
et = time.time()
print(et-st) 


def loadid(path):
    res = []
    file = open(path)
    for f in file:
        res.append(json.loads(f)['newsId'])
    return res
newsid = loadid('data/coreEntityEmotion_test_stage2.txt')

#入袋
bag_S = []
tmp_s = []
last_id = np.nan
i=0
for eid, score in zip(select_ent_id, nt_score):
    if last_id != eid[0]:
        last_id = eid[0]
        if last_id!=newsid[i]:
            bag_S.append([])
            i+=1
        i+=1
        bag_S.append(tmp_s)
        tmp_s = [[eid[1],score]]
    else:
        tmp_s.append([eid[1],score])
bag_S.append(tmp_s)
bag_S = bag_S[1:]
len(bag_S)


def test(bag_S, process_num):
    test_file = '../data/coreEntityEmotion_test_stage2.txt'
    test_file = open(test_file)
    selfs = bag_S
    # feature_names = ["tfidf","tfidfindex","iit","word_dis","seg","text_rank","trindex",
    #                  "tf","tf_bi_count","tfindex","cos","eud","ner_len","first",] #"last","has_num","has_en"

    # print(pd.DataFrame({'column': feature_names,'importance': ents_model.feature_importance(),}).sort_values(by='importance',
    #                                                                                                               ascending=False))
    
    stopwordfile = open("./data/stopwords.txt")
    stopwords = set()
    for stp in stopwordfile:
        stopwords.add(re.sub('\n','',stp))

    count = 0
    has_num = 0
    tal_num = 0
    resX = []
    resY = []
    starttflag = 0
    
    tmp = {}
    newalre=[]
    with open("./results_final/result_"+str(process_num)+".txt","w",encoding="utf-8") as w:
        for news, score in zip(test_file, selfs):
            count+=1
    #         if count >40000:
    #             break
            if count >= 0:
                news = json.loads(news)
                ents = []
                emos = []
                #预测实体
                ent_predict_result = []
                for s in score:
                    ent_predict_result.append([s[0],s[1]])
                ent_predict_result.sort(key=lambda x: x[1], reverse = True)

                 #筛选实体
                c = 0
                try:
                    tops = ent_predict_result[0][1]
                except:
                    pass
                for pred in ent_predict_result:
                    if c == 0:
                        if  pred[1] < 0:
                            break
                    if c == 1:
                        if pred[1] < tops*0.5 or pred[1] < 0.28:
                            break
                    if c == 2:
                        if pred[1] < tops*0.5 or pred[1]< 0.30:
                            break    
                    if c == 3:
                        break
                    flag = 0
                    if pred[0].isdigit():#排除纯数字
                        continue
                    if len(re.sub(r'[0-9]+[年月日亿万千百十个元]', '', pred[0])) == 0:
                        continue
#                     for i in range(len(ents)):
#                         if pred[0] in stopwords:
#                             flag = 1
#                         if re.sub('[\u4e00-\u9fff]+','',ents[i]) == '':
#                             if pred[0] in ents[i] and  len(ents[i])>len(pred[0]):
#                                 flag = 1
#                             if ents[i] in pred[0] and len(pred[0])>len(ents[i]):
#                                 flag = 0
#                                 for i in range(len(ents)): #去除一样的
#                                     if pred[0] == ents[i]:
#                                         flag = 1
#                                 if flag == 0:
#                                     ents[i] = pred[0]
#                                     flag = 1
                    if flag == 0:
                        ents.append(pred[0])
                        c += 1

                ents= sorted(set(ents),key=ents.index)
                for e in ents:
                    if e in tmp:
                        tmp[e] += 1
                    else:
                        tmp[e] = 1
                #预测情感
                for ent in ents:
                    emos.append("POS")
                newalre.append(ents)
#                 print(ent_predict_result[:6])
#                 print(ents,news['title'])
#                 print()
                w.write(news['newsId'] + '\t' + ','.join(ents) + '\t' + ','.join(emos) + '\n')
        c1 = 0
    for n in newalre:
        if len(n) == 0:
            c1 +=1
    print('0 =',c1)
    c1 = 0
    for n in newalre:
        if len(n) == 1:
            c1 +=1
    print('1 =',c1)
    c1 = 0
    for n in newalre:
        if len(n) == 2:
            c1 +=1
    print('2 =',c1)
    c1 = 0
    for n in newalre:
        if len(n) == 3:
            c1 +=1
    print('3 =',c1)
    c1 = 0
    for n in newalre:
        if len(n) > 3:
            c1 +=1
    print('>3 =',c1)
    c1 = 0
    for n in newalre:
        if len(n) == 4:
            c1 +=1
    print('4 =',c1)       
    print("done")
#     print(sorted(tmp.items(),key=lambda item: item[1],reverse=True))

test(bag_S, '0.598')#















