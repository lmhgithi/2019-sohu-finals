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
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec

#文件
# train_file = 'data/coreEntityEmotion_train.txt'
# test_file = 'data/coreEntityEmotion_test_stage1.txt'
# train_file = open(train_file)
# test_file = open(test_file)
# fea_ents = feature_ents(0)
train_ners = load('data/final_train_cut_v3.joblib')
test_ners = load('data/final_test_cut_v3.joblib')

process_num = 0
process_news = []
result_csv = []

texts = []
for ners in tqdm(test_ners):
    texts.append(ners)
for ners in tqdm(train_ners):
    texts.append(ners)
print("开始")
w2vmodel = Word2Vec(texts, size=100, window=5, min_count=1, iter=10, workers=2)
w2vmodel.save("./features/final_word2vec_bert_all.model")
print("done1")
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
d2vmodel = Doc2Vec(documents, vector_size=100, window=2, min_count=1, epochs=10, workers=2)

d2vmodel.save("./features/final_doc2vec_bert_all.model")
print("done2")


# w2vmodel.most_similar('政策')