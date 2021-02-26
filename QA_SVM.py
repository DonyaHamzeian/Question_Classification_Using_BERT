#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, nltk
import gensim
import codecs
from sner import Ner
import spacy
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.internals import find_jars_within_path
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
import spacy
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import fbeta_score, accuracy_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


f_train = open('train5.txt', 'r+')
f_test = open('test.txt', 'r+')

train = pd.DataFrame(f_train.readlines(), columns = ['question'])
test = pd.DataFrame(f_test.readlines(), columns = ['question'])


# In[3]:


train['qType'] = train.question.apply(lambda x: x.split(' ', 1)[0])
train['question'] = train.question.apply(lambda x: x.split(' ', 1)[1])
train['coarse'] = train.qType.apply(lambda x: x.split(':')[0])
test['qType'] = test.question.apply(lambda x: x.split(' ', 1)[0])
test['question'] = test.question.apply(lambda x: x.split(' ', 1)[1])
test['coarse'] = test.qType.apply(lambda x: x.split(':')[0])


# In[4]:


train.head()


# In[5]:


nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer 
# from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize


# In[6]:


nlp = spacy.load("en_core_web_sm")


# In[13]:


# preprocess
train['prep'] = [ re.sub(pattern='[^a-zA-Z0-9]',repl=' ', string = x.lower()) for x in train['question']]
test['prep'] =  [ re.sub(pattern='[^a-zA-Z0-9]',repl=' ', string = x.lower()) for x in test['question']]


# In[38]:


wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']


# In[43]:


# get features for train data
all_ner = []
all_tag = []
heads = []
for row in train['prep']:
    present_ner = []
    present_tag = []
    doc = nlp(row)
    for i, tok in enumerate(doc):
        if i==0:
            if str(tok) not in wh_words:
                heads.append('None')
            else:
                heads.append(tok)
        if tok.tag_!= '':
            present_tag.append(tok.tag_)
    for ent in doc.ents:
        if ent.label_ !='':
            present_ner.append(ent.label_)
    all_ner.append(' '.join(present_ner))
    all_tag.append(' '.join(present_tag))


# In[41]:


# get features for test data
all_ner_test = []
all_tag_test = []
heads_test = []
for row in test['prep']:
    present_ner = []
    present_tag = []
    doc = nlp(row)
    for i, tok in enumerate(doc):
        if i==0:
            if str(tok) not in wh_words:
                heads_test.append('None')
            else:
                heads_test.append(tok)
        if tok.tag_!= '':
            present_tag.append(tok.tag_)
    for ent in doc.ents:
        if ent.label_ !='':
            present_ner.append(ent.label_)
    all_ner_test.append(' '.join(present_ner))
    all_tag_test.append(' '.join(present_tag))


# In[61]:


count_vec_ner = CountVectorizer().fit(all_ner)
ner_ft = count_vec_ner.transform(all_ner)
ner_test_ft = count_vec_ner.transform(all_ner_test)

count_vec_tag = CountVectorizer().fit(all_tag)
tag_ft = count_vec_tag.transform(all_tag)
tag_test_ft = count_vec_tag.transform(all_tag_test)

count_vec_tok = CountVectorizer(stop_words = 'english', min_df = 5).fit(train['prep'])
tok_ft = count_vec_tok.transform(train['prep'])
tok_test_ft = count_vec_tok.transform(test['prep'])


# In[62]:


train['head_chunk']= heads
test['head_chunk']= heads_test
heads_dummies = train.append(test).head_chunk.str.get_dummies()
head_ft = heads_dummies[0:len(train)]
head_test_ft = heads_dummies[len(train):]


# In[63]:


# prepare data for training
x_all_ft_train = hstack([ner_ft, tag_ft, tok_ft, head_ft])
x_all_ft_train = x_all_ft_train.tocsr()

x_all_ft_test = hstack([ner_test_ft, tag_test_ft, tok_test_ft, head_test_ft])
x_all_ft_test = x_all_ft_test.tocsr()



# In[65]:


x_all_ft_train.shape


# In[64]:


x_all_ft_test.shape


# In[69]:


# model training SVM


model_svm = svm.LinearSVC()
model_svm.fit(x_all_ft_train, train['coarse'].values)


# In[71]:


# evaluate model
preds = model_svm.predict(x_all_ft_test)
print('svm = {}'.format(accuracy_score(test['coarse'].values, preds)))


# In[ ]:




