import os
from nltk.parse import stanford
from snownlp import SnowNLP
import jieba
import pandas as pd
from category_encoders import *

pos = '这个苹果好甜呀'
neg = '我吃到的这个梨特别酸'
oth = '这个苹果真难吃'
# text = 'But parsers can also be used to derive other kinds of tree structure, such as morphological trees and discourse structures.'

os.environ['STANFORD_PARSER'] = r"/data/wangm/GLT-fastCFs/stanford-parser-full-2020-11-17/stanford-parser.jar"
os.environ['STANFORD_MODELS'] = r"/data/wangm/GLT-fastCFs/stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar"
parser = stanford.StanfordDependencyParser(model_path=r"/data/wangm/GLT-fastCFs/stanford-parser-full-2020-11-17/edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz")

pos_s = jieba.lcut(pos)
neg_s = jieba.lcut(neg)
oth_s = jieba.lcut(oth)

# s = SnowNLP(text)
# text = s.words
# res = list(parser.parse(text))
# for row in res[0].triples():
#     print(row)
pos_dict = {}
pos_res = list(parser.parse(pos_s))
for row in pos_res[0].triples():
    key1 = row[0][1]+row[1]
    value1 = row[0][0]
    key2 = row[1] + row[2][1]
    value2 = row[2][0]
    pos_dict[key1] = value1
    pos_dict[key2] = value2
    pass

neg_dict = {}
neg_res = list(parser.parse(neg_s))
for row in neg_res[0].triples():
    key1 = row[0][1] + row[1]
    value1 = row[0][0]
    key2 = row[1] + row[2][1]
    value2 = row[2][0]
    neg_dict[key1] = value1
    neg_dict[key2] = value2
    pass
oth_dict = {}
oth_res = list(parser.parse(oth_s))
for row in oth_res[0].triples():
    key1 = row[0][1] + row[1]
    value1 = row[0][0]
    key2 = row[1] + row[2][1]
    value2 = row[2][0]
    oth_dict[key1] = value1
    oth_dict[key2] = value2
    pass

pos_df = pd.DataFrame([pos_dict])
neg_df = pd.DataFrame([neg_dict])
oth_df = pd.DataFrame([oth_dict])

df = pd.concat([pos_df,neg_df,oth_df],axis=0).reset_index()
df["label"] = [1,0,0]

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

encoder = TargetEncoder().fit(X,y)
ndata = encoder.transform(X)

print(ndata)
