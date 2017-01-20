#coding=utf-8
import pandas as pd
def create_Bayes_prob(train,col_names):
    T_prob = sum(train.label)/float(len(train))
    F_prob = 1 - T_prob
    Bayes_prob = {0:{},1:{}}
    for i in [0,1]:
        train_use = train[train.label == i]
        L = float(len(train_use))
        for col in col_names:
            category = train[col].unique()
            Bayes_prob[i][col] = {}
            for j in category:
                Bayes_prob[i][col][j] = len(train_use[train_use[col] == j]) / L#可以改进进行平滑处理
    return Bayes_prob,T_prob,F_prob

def Bayes_pred(test,col_names,T_prob,F_prob,Bayes_prob={}):
    pred = {}
    F = []
    T = []
    for i in range(len(test)):
        line = test.irow(i)
        p = 1.0
        for col in col_names:
            p = p*Bayes_prob[0][col][line[col]]
        p = p*F_prob
        F.append(p)

        p = 1.0
        for col in col_names:
            p = p*Bayes_prob[1][col][line[col]]
        p = p*T_prob
        T.append(p)
    for i in range(len(test)):
        s = T[i] + F[i]
        T[i] = round(T[i]/s,1)
        F[i] = round(F[i]/s,1)
    pred['0'] = F
    pred['1'] = T
    return pred

def write_to_csv(pred,test,path):
    result = pd.DataFrame()
    result['probability'] = pred
    result.index  = test.index
    result.to_csv(path)











