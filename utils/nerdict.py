import jieba
import re
import json
import csv
import pkuseg
import time
import os
from tqdm import tqdm
class jieba_ner():
    def __init__(self, flag):
        self.train_file_path = "../data/coreEntityEmotion_train.txt"
        self.get_stopwords()
        self.jieba = jieba
        self.jieba.set_dictionary('/home/bigdata/anaconda3/lib/python3.7/site-packages/jieba/dict.txt.big')
        #直接用切好的词了
        self.load_user_dict(flag)
        
    def get_jieba(self):
        return self.jieba
    
    '''def load_all_dict(self, flag):
        print("loading all user nerdict...")
        st = time.time()
        count = 0
        work_dir = '/home/ftp/Jupyter_Workplace/competition/coreEntityEmotion_baseline/data/user_nerdict'
        for parent, dirnames, filenames in os.walk(work_dir,  followlinks=True):
            for filename in tqdm(filenames):
                if filename == '实体词典.txt' and flag == 3:
                    continue
                file_path = os.path.join(parent, filename)
                self.jieba.load_userdict(file_path)
                count += 1
        et = time.time()
        print(count,"files loaded")
        print("all user nerdict is loaded, time use is:", et-st)'''   
        
    def load_user_dict(self, flag):
        print("loading user nerdict...")
        st = time.time()
        if flag != -1:
            self.jieba.load_userdict("./data/user_nerdict/FIFA.txt")
            self.jieba.load_userdict("./data/user_nerdict/NBA.txt")
            self.jieba.load_userdict("./data/user_nerdict/origin_zimu.txt")
            self.jieba.load_userdict("./data/user_nerdict/person.txt")
            self.jieba.load_userdict("./data/user_nerdict/val_keywords.txt")
            self.jieba.load_userdict("./data/user_nerdict/出现的作品名字.txt")
            self.jieba.load_userdict("./data/user_nerdict/创造101.txt")
            self.jieba.load_userdict("./data/user_nerdict/动漫.txt")
            self.jieba.load_userdict("./data/user_nerdict/手机型号.txt")
            self.jieba.load_userdict("./data/user_nerdict/明星.txt")
            self.jieba.load_userdict("./data/user_nerdict/显卡.txt")
            self.jieba.load_userdict("./data/user_nerdict/歌手.txt")
            self.jieba.load_userdict("./data/user_nerdict/流行歌.txt")
            self.jieba.load_userdict("./data/user_nerdict/漫漫看_明星.txt")
            self.jieba.load_userdict("./data/user_nerdict/电影.txt")
            self.jieba.load_userdict("./data/user_nerdict/电视剧.txt")
            self.jieba.load_userdict("./data/user_nerdict/百度明星.txt")
            self.jieba.load_userdict("./data/user_nerdict/百度热点人物+手机+软件.txt")
            self.jieba.load_userdict("./data/user_nerdict/篮球.txt")
            self.jieba.load_userdict("./data/user_nerdict/网络流行新词.txt")
            self.jieba.load_userdict("./data/user_nerdict/美食.txt")
            self.jieba.load_userdict("./data/user_nerdict/自定义词典.txt")
            self.jieba.load_userdict("./data/user_nerdict/足球.txt")
            self.jieba.load_userdict("./data/user_nerdict/nerDict.txt")
            self.jieba.load_userdict("./data/user_nerdict/bertcibiao.txt")
            self.jieba.load_userdict("./data/user_nerdict/final_bertcibiao.txt")
            
            if flag == 1:
                self.jieba.load_userdict("./data/user_nerdict/实体词典.txt")
                #self.jieba.load_userdict("./data/user_nerdict/测试集实体词典.txt")
            elif flag == 0:
                self.jieba.load_userdict("./data/user_nerdict/实体词典2.txt")
        et = time.time()
        print("user nerdict is loaded, time use is:", et-st)
        
    def get_stopwords(self):
        stopwordfile = open("./data/stopwords.txt")
        self.stopwords = set()
        for stp in stopwordfile:
            self.stopwords.add(re.sub('\n','',stp))
            
    def returnstpwords(self):
        return self.stopwords
        
    def is_json(self, news):
        try:
            json.loads(news)
            return True
        except:
            return False
            
    #全模式cut单个news用于预测
    def jieba_cut(self, news):

        title = news['title']
        content = news['content']
        nerdict = set()
        ''' 惊了，做正则覆盖率降低了
        sentences = []
        for seq in re.split(r'[\n。，、：‘’“""”？！?!]', title+"，"+content):
            #seq = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", "",seq)
            if len(seq) > 2:
                sentences.append(seq)
        '''       
                
        #for seq in sentences:
        ner_list = self.jieba.cut(title+"，"+content)  #cut_for_search   cut_all = False
        for ner in ner_list:
            if len(ner) > 1:
                if ner not in self.stopwords:  #去停用词
                    nerdict.add(ner)
        
        return nerdict #set
    
        