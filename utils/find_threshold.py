from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm
# 定义寻找最佳分割阈值的函数
def threshold_search(y_true, y_pred, y_probability, n):
    # 预存最佳的阈值和f1_score
    best_score = -1
    for threshold1 in [i * 0.1 for i in range(0,1)]:
        for threshold2 in [i * 0.1 for i in range(2,5)]:
            for threshold3 in [i * 0.1 for i in range(2,5)]:
                for threshold4 in [i * 0.1 for i in range(1,4)]:
                    for threshold5 in [i * 0.1 for i in range(1,5)]:
#                        计算每一个阈值的f1得分
                        c = 0
                        pred_metric = []
                        true_metric = []
                        for true, pred, proba in zip(y_true,y_pred,y_probability):
                            ents = []
                            Y_ents = []
                            c = 0
                            try:
                                tops = proba[0]
                            except:
                                tops = 1
                            for ped, prba in zip(pred, proba):
                                if c == n:
                                    break
                                if c==0:
                                    if  prba < threshold1:
                                        break
                                if c==1:
                                    if prba< tops*threshold2 or prba < threshold4:
                                        break
                                if c==2:
                                    if prba <  tops*threshold3 or prba< threshold5:
                                        break
                                ents.append(ped)
                                c += 1 
                            for y in true:
                                Y_ents.append(y)
                            #模型评估
                            for pred in ents:
                                if pred in Y_ents: #TP
                                    pred_metric.append(1)
                                    true_metric.append(1)
                                else:              #FP
                                    pred_metric.append(1) 
                                    true_metric.append(0)
                            for y in Y_ents:       #FN
                                if y not in ents:
                                    pred_metric.append(0)
                                    true_metric.append(1)
                        score = f1_score(true_metric, pred_metric)
                        if score > best_score:
                            best_threshold1 = threshold1
                            best_threshold2 = threshold2
                            best_threshold3 = threshold3
                            best_threshold4 = threshold4
                            best_threshold5 = threshold5
                            best_score = score
                            '''r = recall_score(true_metric, pred_metric)
                            p = precision_score(true_metric, pred_metric)'''
        print('tmp best f1-score:',best_score)
    search_result = {'best_f1_socre': best_score}#, 'precision':p,'recall':r
    thresholds = {
        'best_threshold1': best_threshold1,
        'best_threshold2': best_threshold2,
        'best_threshold3': best_threshold3,
        'best_threshold4': best_threshold4 ,
        'best_threshold5': best_threshold5
    }
    print(search_result)
    print(thresholds)
