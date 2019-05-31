## sohu2019校园算法大赛
> 队名：我想去北京。\
> 名次 初赛6，决赛10。\
> 这只是lightgbm单模的代码，实体初赛分数大概至少56+。详细的方案介绍在ppt里。\
> [比赛网址](https://biendata.com/competition/sohu2019/)

- 运行run.sh就可以直接按顺序运行全部代码，**但请慎重，在步骤中会开启多进程并且耗费非常长的时间，建议按顺序略读代码即可。**
- 事情比较多，代码只是稍微整理了一下，没跑过，如果跑不通可以提issue

环境：
- python3.7
- linux
- 内存128gb
- 依赖包：tqdm、pandas、numpy、lightgbm、joblib、re、sklearn、collections、imblearn、jieba、gensim、scipy

各文件及其作用介绍：
1. /data 存放词典，停用词，以及分词等数据
2. /features 存放计算好的特征，里面的all_classes.csv来源于朱帅计算的类别特征
3. /models 存放训练好的模型
4. /results 存放预测的结果
5. /utils
   - features_ents.py 特征计算代码，所有特征计算代码都在这个文件下
   - nerdict.py 一些分词时用的代码，有加载词典、分词等方法，被features_ents.py调用
   - find_threshold.py 阈值搜索代码，设置了5个阈值，可以在验证集找到合适的阈
值，并且在预测时使用，可以根据模型预测分数来控制实体数目
6. 其他的代码
   - features_cal.py调用features_ents计算特征（多进程，两个16进程，所以慎重运行），特征存入/features
   - cut.py 分词，结果存入data/
   - tfidf.py 训练tfidf模型，存到models/
   - d2v_w2v.py 训练doc2vec和word2vec模型，模型存到models/
   - ners_select_features.py 计算概率相关特征 ，结果存入/features
   - train.py 选择特征，并训练模型，结果存入/models
   - test.py  预测，得到结果并输入到，结果存入/results_final

- 如果对大家有帮助，就帮忙点个:star:
