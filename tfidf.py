import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from joblib import load, dump
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

train_ners = load('data/final_train_cut_v3.joblib')
test_ners = load('data/final_test_cut_v3.joblib')

nerCorpus = []
for ners in train_ners:
    nerCorpus.append(' '.join(ners))
for ners in test_ners:
    nerCorpus.append(' '.join(ners))
    
tfIdf = TfidfVectorizer()
tfIdf.fit(nerCorpus)

dump(tfIdf, 'features/final_nerTfIdf_bert_all.joblib',compress = 0)