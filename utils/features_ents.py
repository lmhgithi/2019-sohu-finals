from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from utils.nerdict import jieba_ner
from joblib import dump,load
from tqdm import tqdm
import time
import json
# from textrank4zh import TextRank4Keyword, TextRank4Sentence
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
import math
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import scipy.stats as stats
import jieba.posseg as pseg

class feature_ents():
    def __init__(self, flag):
        self.pseg = pseg
        self.train_file_path = "./data/coreEntityEmotion_train.txt"
        #self.holdout_train_file_path = "./data/train_73holdout.txt"
        self.text_rank = TextRank4Keyword()
        self.jiebaner = jieba_ner(flag)  #flag = 0 1 2 3
        
        self.jieba = self.jiebaner.get_jieba()
        self.d2v_model = Doc2Vec.load("features/final_doc2vec_bert_all.model")
        self.w2v_model = Word2Vec.load("features/final_word2vec_bert_all.model")
        self.wv = self.w2v_model.wv
        self.nertype_encoder = {}
        self.tfIdf = load('features/final_nerTfIdf_bert_all.joblib')
        self.featureName = self.tfIdf.get_feature_names()
        self.stopwords = self.jiebaner.returnstpwords()
                    
        
    def set_ners(self, ners):
        self.ners = []
        for ner in ners:
            if re.sub('.','',ner).isdigit():
                continue
            if ner in self.stopwords:
                continue
            self.ners.append(str(ner))
            
    def cut_ners(self, news):
        self.ners = set(self.jiebaner.jieba_cut(news))
        
    def get_ners(self):
        return self.ners
    
    def getjieba(self):
        return self.jieba
    
    
    # TextRankMerge_4zh 用来做词语融合，发现长词
    def TextRankMerge_4zh(self, news):
        TextRankScore = {}
        
        title = news['title']
        content = news['content']
        sentences = []
        '''
        for seq in re.split(r'[\n。，、：‘’“""”？！?!《》]', title+" "+content):
            seq = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "",seq)
            if len(seq) > 2:
                sentences.append(seq)'''
        #text = " ".join(sentences)
        text = title + ' '+content
        
        self.text_rank.analyze(text=text, lower=True, window=2)
        #TextRankScore = {}
        #max_score = 0
        for item in self.text_rank.get_keywords(200, word_min_len=4):
            if item.word not in self.ners:
                self.ners.add(item.word)
        for phrase in self.text_rank.get_keyphrases(keywords_num = 200, min_occur_num = 2):
            if phrase not in self.ners:
                self.ners.add(phrase)
        #return TextRankScore
        
    
    # textrank分数 和 tfidf分数
    def get_tr_Score(self, news):
        title = news['title']
        content = news['content']
        article = title+ '，'+content
        text_rank = {}
        trindex = {}
        all_pro = ['a', 'ad', 'ag', 'an', 'b', 'c', 'd',
            'df', 'dg', 'e', 'eng', 'f', 'g', 'h',
            'i', 'j', 'k', 'l', 'm', 'mg', 'mq',
            'n', 'ng', 'nr', 'nrfg', 
            'nrt', 'ns', 'nt', 'nz',
            'o', 'p', 'q', 'r', 'rg',
            'rr', 'rz', 's', 't', 'tg',
            'u', 'ud', 'ug', 'uj', 'ul', 'uv', 'uz',
            'v', 'vd', 'vg', 'vi',
            'vn', 'vq', 'w', 'x', 
            'y', 'z', 'zg']
        for name, score in self.jieba.analyse.textrank(article, topK=1200, withWeight=True, allowPOS=all_pro):
            text_rank[name] = score
        sorted(text_rank,reverse=True)
        rank = 0
        for ner in text_rank:
            trindex[ner] = rank
            rank += 1
            
        return text_rank, trindex
    
    # Tfidf分数 sklearn 和tfidfindex
    def get_tfidf_Score(self):
        tfidfindex = {}
        tfIdfNameScore = {}
        tfIdf = self.tfIdf
        tfIdfFeatures = tfIdf.transform([' '.join(self.ners)])
        tfIdfScores = tfIdfFeatures.data
        # normalize
        tfIdfScoresNorm = [[]]
        if len(tfIdfScores) > 0:
            tfIdfScoresNorm = normalize([tfIdfScores], norm='max')
        
        for name, score in zip(tfIdfFeatures.indices, tfIdfScoresNorm[0]):
            tfIdfNameScore[self.featureName[name]] = score
        NameScore = sorted(tfIdfNameScore.items(), key=lambda item: item[1], reverse=True)
        rank = 0
        for NS in NameScore:
            tfidfindex[NS[0]] = rank
            rank += 1
        
        return tfIdfNameScore, tfidfindex
    
    # word2vec 和 doc2vec的余弦距离和欧式距离
    def w2v_d2v(self, vec1, vec2):
        npvec1, npvec2 = np.array(vec1), np.array(vec2)
        
        Cosine = npvec1.dot(npvec2)/(math.sqrt((npvec1**2).sum()) * math.sqrt((npvec2**2).sum()))
        Euclidean = math.sqrt(((npvec1-npvec2)**2).sum())
        return Cosine, Euclidean
               
    
    #for_ner操作的特征
    def for_ner(self, news):
        first = {} #第一句
        last = {}  #最后一句
        #mid = {}  #中间
        freq_ent = {} #词被选为实体的次数（在训练集统计）
        ners_select_per = {} #词被选为实体的概率 （在训练集统计）
        
        iit = {} #是否在标题
        word_dis = {} #词跨度
        word_dis_index = {}
        tf = {} #词频
        tf_bi_len = {} #词频/文章长度
        cos = {}   #w2v和d2c的余弦距离
        eud = {}    #w2v和d2c的欧式距离
        has_num = {}  #有数字
        has_en = {} #有英文
        tfindex = {}  #词频排名
        baohan = {} #是否被包含
        
        pattern_num = re.compile('[0-9]+')
        pattern_en = re.compile('[a-zA-z]+')
        title = news['title']
        content = news['content']
        article = title+ ' '+content
        maxd = 0
        
        for ner in self.ners:
            baohan[ner] = 0
            for n in self.ners:
                if ner in n and len(ner)<len(n):
                    baohan[ner] = 1
                    break
            
            if pattern_num.findall(ner):
                has_num[ner] = 1
            else:
                has_num[ner] = 0
            if pattern_en.findall(ner):
                has_en[ner] = 1
            else:
                has_en[ner] = 0
            
            if ner in self.wv:
                cos[ner], eud[ner] = self.w2v_d2v(self.wv[ner], self.doc_vec)
            
            tf[ner] = article.count(ner)
            tf_bi_len[ner] = tf[ner] / (len(self.ners)+1)
            '''if ner in self.ners_select_per:
                ners_select_per[ner] = self.ners_select_per[ner]
                
            if ner in self.freq_ent:
                freq_ent[ner] = self.freq_ent[ner]'''
            
            d = content.rfind(ner)-content.find(ner)
            word_dis[ner] = d/len(content)
            
            if ner in title:
                iit[ner] = 1
            else:
                iit[ner] = 0
            if ner in news["content"].split("。")[0]:
                first[ner] = 1
            else:
                first[ner] = 0
            if len(news["content"].split("。"))>1:
                if ner in news["content"].split("。")[-2]:
                    last[ner] = 1
                else:
                    last[ner] = 0
            else:
                last[ner] = 0
        #tfindex
        tfns = sorted(tf.items(), key=lambda item: item[1], reverse=True)
        rank = 0
        tmprank = 0
        tmprankscore = np.nan
        for ns in tfns:
            if ns[1] != tmprankscore:
                tmprank = rank
                tmprankscore = ns[1]
                tfindex[ns[0]] = rank
                rank += 1
            else:
                tfindex[ns[0]] = tmprank
                rank += 1
                
        wdns = sorted(word_dis.items(), key=lambda item: item[1], reverse=True)
        rank = 0
        tmprank = 0
        tmprankscore = np.nan
        for ns in wdns:
            if ns[1] != tmprankscore:
                tmprank = rank
                tmprankscore = ns[1]
                word_dis_index[ns[0]] = rank
                rank += 1
            else:
                word_dis_index[ns[0]] = tmprank
                rank += 1
         
        '''topeud = 0
        for e in eud:
            if eud[e] > topeud:
                topeud = eud[e]
        if topeud != 0:
            for e in eud:
                eud[e] = eud[e]/topeud'''
       
        '''#中间
        if len(sentence) > 2:
            for sent in sentence[1:len(sentence)-1]:
                for ner in self.ners:
                    if ner in sent:
                        mid[ner] = 1'''
        return word_dis, word_dis_index, iit, tf, tf_bi_len, cos, eud, has_num, has_en, first, last, tfindex, baohan

    #共现矩阵信息
    def gongxian(self, news):
        ske = {}   # 偏度
        var_gongxian = {}  #方差
        kurt_gongxian = {}  #峰度
        diff_min_gongxian = {}
    
        sentences = [news['title']]
        for seq in re.split(r'[\n。？！?!.]', news['content']):
            # 如果开头不是汉子、字母或数字，则去除
            seq = re.sub(r'^[^\u4e00-\u9fa5A-Za-z0-9]+', '', seq)
            # 去除之后，如果句子不为空，则添加进句子中
            if len(seq) > 0: 
                sentences.append(self.jieba.lcut(seq))
        num_tokens = len(self.ners)
        words_list = list(self.ners)
        arr = np.zeros((num_tokens, num_tokens))
        # 得到共现矩阵
        for i in range(num_tokens):
            for j in range(i+1, num_tokens):
                count = 0 
                for sentence in sentences:
                    if words_list[i] in sentence and words_list[j] in sentence:
                        count += 1 
                arr[i, j] = count
                arr[j, i] = count
                
        skearr = stats.skew(arr)
        for i in range(num_tokens):
            ske[words_list[i]] = skearr[i]
            var_gongxian[words_list[i]] = np.var(arr[i])
            kurt_gongxian[words_list[i]] = stats.kurtosis(arr[i])
            diff_sim = np.diff(arr[i])
            if len(diff_sim) > 0:
                diff_min_gongxian[words_list[i]] = np.min(diff_sim)
      
        return ske, var_gongxian, kurt_gongxian, diff_min_gongxian
    
        
    # 计算相似度矩阵以及统计信息
    def corr(self):
        ## sim_tags_arr: 初始化候选词相似度矩阵
        mean_sim_tags = {}
        skew_sim_tags = {}
        kurt_sim_tags = {}
        diff_mean_sim_tags = {}
        diff_skew_sim_tags = {}
        diff_kurt_sim_tags = {}
        
        num_tokens = len(self.ners)
        words_list = list(self.ners)
        sim_tags_arr = np.zeros((num_tokens, num_tokens))

        for i in range(num_tokens):
            for j in range(i, num_tokens):
                sim_tags_arr[i][j] = 0 if (words_list[i]  not in self.wv.vocab or words_list[j] not in self.wv.vocab) else self.wv.similarity(words_list[i], words_list[j])
                if i != j:
                    sim_tags_arr[j][i] = sim_tags_arr[i][j]
        # 计算单词相似度矩阵的统计信息
        tmp = {}
        ## 相似度平均值
        tmp['mean_sim_tags'] = np.mean(sim_tags_arr, axis=0)
        ## 相似度矩阵的偏度
        tmp['skew_sim_tags'] = stats.skew(sim_tags_arr, axis=0)
        ## 相似度矩阵的峰值
        tmp['kurt_sim_tags'] = stats.kurtosis(sim_tags_arr, axis=0)
        ## 相似度矩阵的差分均值
        tmp['diff_mean_sim_tags'] = np.mean(np.diff(sim_tags_arr, axis=0), axis=0)
        ## 相似度矩阵差分的偏度
        tmp['diff_skew_sim_tags'] = stats.skew(np.diff(sim_tags_arr, axis=0), axis=0)
        ## 相似度矩阵差分的峰度
        tmp['diff_kurt_sim_tags'] = stats.kurtosis(np.diff(sim_tags_arr, axis=0), axis=0)
        
        for i in range(num_tokens):
            mean_sim_tags[words_list[i]] = tmp['mean_sim_tags'][i]
            skew_sim_tags[words_list[i]] = tmp['skew_sim_tags'][i]
            kurt_sim_tags[words_list[i]] = tmp['kurt_sim_tags'][i]
            diff_mean_sim_tags[words_list[i]] = tmp['diff_mean_sim_tags'][i]
            try:
                diff_skew_sim_tags[words_list[i]] = tmp['diff_skew_sim_tags'][i]
                diff_kurt_sim_tags[words_list[i]] = tmp['diff_kurt_sim_tags'][i]
            except:
                print("fea_ents第338行报错了")
                diff_skew_sim_tags[words_list[i]] = np.nan
                diff_kurt_sim_tags[words_list[i]] = np.nan
            
        return mean_sim_tags, skew_sim_tags, kurt_sim_tags, diff_mean_sim_tags, diff_skew_sim_tags, diff_kurt_sim_tags

     #  词性
    def get_NerType(self, ner):
        try:
            nertype = next(self.pseg.cut(ner)).flag
        except:
            nertype = 'wz'
        if nertype in self.nertype_encoder:
            return self.nertype_encoder[nertype]
        else:
            self.nertype_encoder[nertype] = len(self.nertype_encoder)
            return self.nertype_encoder[nertype]
        

    
    # 每次重复所有特征太慢了，这里只计算新特征，然后拼起来
    def combine_new_features(self, news):
        ske, var_gongxian, kurt_gongxian, diff_min_gongixna = self.gongxian(news)
        mean_sim_tags, skew_sim_tags, kurt_sim_tags, diff_mean_sim_tags, diff_skew_sim_tags, diff_kurt_sim_tags = self.corr()
        features = []
        result = pd.DataFrame()
        for ner in self.ners:
            x = {}
            tmp = []      
            x['ske']=np.nan
            x['var_gongxian']=x['kurt_gongxian']=x['diff_min_gongixna']=np.nan
            x['mean_sim_tags']=x['skew_sim_tags']=x['kurt_sim_tags']=x['diff_mean_sim_tags']= x['diff_skew_sim_tags']=x['mean_sim_tags']=np.nan
            
            x['newsid'] = news['newsId']
            x['ner'] = ner
            if ner in ske:
                x['ske'] = ske[ner]
            if ner in var_gongxian:
                x['var_gongxian'] = var_gongxian[ner]
            if ner in kurt_gongxian:
                x['kurt_gongxian'] = kurt_gongxian[ner]
            if ner in diff_min_gongixna:
                x['diff_min_gongixna'] = diff_min_gongixna[ner]
                
            if ner in mean_sim_tags:
                x['mean_sim_tags'] = mean_sim_tags[ner]
            if ner in skew_sim_tags:
                x['skew_sim_tags'] = skew_sim_tags[ner]
            if ner in kurt_sim_tags:
                x['kurt_sim_tags'] = kurt_sim_tags[ner]
            if ner in diff_mean_sim_tags:
                x['diff_mean_sim_tags'] = diff_mean_sim_tags[ner]
            if ner in diff_skew_sim_tags:
                x['diff_skew_sim_tags'] = diff_skew_sim_tags[ner]
            if ner in diff_kurt_sim_tags:
                x['diff_kurt_sim_tags'] = diff_kurt_sim_tags[ner]
            
            tmp.append([x['ske'], x['var_gongxian'], x['kurt_gongxian'], x['diff_min_gongixna'], x['mean_sim_tags'],x['skew_sim_tags'],x['kurt_sim_tags'],x['diff_mean_sim_tags'],x['diff_skew_sim_tags'],x['diff_kurt_sim_tags']])
            
            df=pd.DataFrame(tmp,columns=["ske","var_gongxian","kurt_gongxian","diff_min_gongixna", "mean_sim_tags","skew_sim_tags","kurt_sim_tags","diff_mean_sim_tags","diff_skew_sim_tags","diff_kurt_sim_tags"])
            result=pd.concat([result,df],axis=0)
        return result
    
    # 把特征接到一起
    def combine_features(self, news, ners):
        self.set_ners(ners)
        #self.TextRankMerge_4zh(news)
        self.doc_vec = self.d2v_model.infer_vector(list(self.ners))  #设置本篇文章的doc vec
        word_dis, word_dis_index, iit, tf, tf_bi_len, cos, eud, has_num, has_en, first, last, tfindex, baohan = self.for_ner(news) 
        text_rank, trindex = self.get_tr_Score(news)
        tfidf, tfidfindex = self.get_tfidf_Score()
        features = []
        result = pd.DataFrame()
        for ner in self.ners:
            x = {}
            tmp = []
            x['newsid']=x['ner']=x['tfidf']=x['tfidfindex']= x['iit']=x['word_dis']=x['seg']=x['text_rank']=x['trindex']=x['tf']=x['tf_bi_len']= x['cos']= x['eud']=x['ner_len']=x['first']=x['last']=x["tfindex"]=x["has_num"]=x["has_en"]=x["baohan"]= np.nan
            x['newsid'] = news['newsId']
            x['ner'] = ner
            x['seg'] = self.get_NerType(ner)
            x['ner_len'] = len(ner)
            if ner in tfidf:
                x['tfidf'] = tfidf[ner]
                x['tfidfindex'] = tfidfindex[ner]
            if ner in iit:
                x['iit'] = iit[ner]
            if ner in word_dis:
                x['word_dis'] = word_dis[ner]
                x['word_dis_index'] = word_dis_index[ner]
            if ner in text_rank: 
                x['text_rank'] = text_rank[ner]
                x['trindex'] = trindex[ner]
            if ner in tf:
                x['tf'] = tf[ner]
                x['tf_bi_len'] = tf_bi_len[ner]
                x["tfindex"] = tfindex[ner]
            if ner in cos:
                x['cos'] = cos[ner]
                x['eud'] = eud[ner]
            if ner in first:
                x['first'] = first[ner]
            if ner in last:   
                x['last'] = last[ner]
            if ner in has_num:
                x['has_num'] = has_num[ner]
            if ner in has_en:
                x['has_en'] = has_en[ner]
            x["baohan"] = baohan[ner]
            
            tmp.append([x['newsid'], x['ner'], x['tfidf'], x['tfidfindex'], x['iit'], x['word_dis'],x['word_dis_index'], x['seg'], x['text_rank'],x['trindex'], x['tf'], x['tf_bi_len'], x["tfindex"], x['cos'], x['eud'], x['ner_len'], x['first'], x['last'],x['has_num'],x['has_en'],x["baohan"]])
            
            df=pd.DataFrame(tmp,columns=["newsid","ner","tfidf","tfidfindex","iit","word_dis","word_dis_index","seg",                         "text_rank","trindex","tf","tf_bi_len","tfindex","cos","eud","ner_len","first","last","has_num","has_en","baohan"])
            result=pd.concat([result,df],axis=0)
        return result

    '''
           if ner in freq_ent:   
                x4 = freq_ent[ner]
            if ner in ners_select_per:  
                x5 = ners_select_per[ner]   
            if ner in mid:    
                g = mid[ner]
            p = is_en[ner]     
            q = has_num[ner]'''