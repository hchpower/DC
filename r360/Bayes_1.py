#coding=utf-8
import pandas as pd
from LR_help import KS_score
#import Bayes_help
#reload(Bayes_help)
from Bayes_help import *
from sklearn.cross_validation import train_test_split
user_info_train = pd.read_csv('E:/DC/r360/credit/train/user_info_train.txt',header = None)
user_info_test = pd.read_csv('E:/DC/r360/credit/test/user_info_test.txt',header = None)
# 设置字段（列）名称
col_names = ['userid', 'sex', 'occupation', 'education', 'marriage', 'household']
user_info_train.columns = col_names
user_info_test.columns = col_names
user_info_train.index = user_info_train['userid']
user_info_train.drop('userid',axis = 1,inplace = True)
user_info_test.index = user_info_test['userid']
user_info_test.drop('userid',axis = 1,inplace = True)
#-----------------------------------------------------------------#
target = pd.read_csv('E:/DC/r360/credit/train/overdue_train.txt',header = None)
target.columns = ['userid', 'label']
target.index = target['userid']
target.drop('userid',axis = 1,inplace = True)
train = user_info_train.join(target, how = 'outer')
#-------------------------测试----------------------------------------#
train_X, test_X = train_test_split(train,test_size = 0.2,random_state = 0)
col_names = ['sex', 'occupation', 'education', 'marriage', 'household']
BayesProb,T_prob,F_prob = create_Bayes_prob(train_X,col_names)
pred = Bayes_pred(test_X,col_names,T_prob,F_prob,Bayes_prob=BayesProb)
print 'KS得分' + str(KS_score(test_X['label'],pred['1']))

#-------------------------使用----------------------------------------#
col_names = ['sex', 'occupation', 'education', 'marriage', 'household']
BayesProb,T_prob,F_prob = create_Bayes_prob(train,col_names)
pred = Bayes_pred(user_info_test,col_names,T_prob,F_prob,Bayes_prob=BayesProb)
path = 'E:\\DC\\r360\\credit\\result\\Bayes.csv'
write_to_csv(pred['1'],user_info_test,path)
