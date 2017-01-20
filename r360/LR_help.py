#coding=utf-8
import pandas as pd
import numpy as np
import math
def KS_score(lable,pred):#计算ks得分
    #pred均保留一位小数
    lable_one = float(sum(lable))
    lable_zero = float(len(lable) - lable_one)
    result = pd.DataFrame()
    result['lable'] = lable
    result['pred'] = pred
    result = result.sort(columns='pred')
    pred_value = list(set(pred))
    pred_value.sort()
    f = [[0,0]]#第一个为标签为1，后一个为0
    pos = 0
    for p in pred_value:
        p_lable = result[result['pred'] == p]['lable']
        f.append([sum(p_lable)+f[pos][0],len(p_lable)-sum(p_lable)+f[pos][1]])
        pos = pos + 1
    f = f[1:]
    k = []
    for line in f:
        k.append(math.fabs(line[0]/lable_one - line[1]/lable_zero))
    return max(k)


