import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
import random
import jieba
import os
from string import punctuation
raw_data = pd.read_csv("data/training.1600000.processed.noemoticon.csv" ,names=["情绪", "编号", "日期", "平台","名称", "评论"],encoding = "ISO-8859-1")
test_data = pd.read_csv("data/testdata.manual.2009.06.14.csv" ,names=["情绪", "编号", "日期", "平台","名称", "评论"],encoding = "ISO-8859-1")
add_punc = '{}()%^ >.^ -=&#@'
add_punc = add_punc + punctuation
c = CountVectorizer(stop_words='english')
# raw_data = pd.concat([raw_data, test_data], ignore_index=True)
X = raw_data['评论']
y_dict = {0: 0,  4: 1}
y = raw_data['情绪'].map(y_dict)
model = c
clf_model = LogisticRegression(max_iter=1000)
X_c = model.fit_transform(X)
# X_c, X_test = X_c[0:-360], X_c[-360,0]
print('# features: {}'.format(X_c.shape [1]))
X_train , X_valid, y_train , y_valid = train_test_split(X_c , y, test_size=0.1,random_state =0)
print('# train records: {}'.format(X_train.shape[0]))
print('# valid records: {}'.format(X_valid.shape[0]))
clf = clf_model.fit(X_train , y_train)
acc = clf.score(X_valid , y_valid)
print('valid Accuracy: {}'.format(acc))

x_test = test_data['评论']
X_test = model.transform(x_test)
y_test = test_data['情绪'].map(y_dict)

acc = clf.score(X_test , y_test)
print('test Accuracy: {}'.format(acc))