#!/usr/bin/env python
# coding: utf-8

# In[92]:


import re
import numpy as np
import pandas as pd
import string
# plotting
import seaborn as sns
import matplotlib.pyplot as plt
# nltk
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from collections import defaultdict
#Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
 
warnings.filterwarnings(action = 'ignore')
 
import gensim
from gensim.models import Word2Vec

# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score


# ## Loading the Dataset

# In[70]:


DATASET_COLUMNS = ['claims_id','claims','status']
DATASET_ENCODING = "ISO-8859-1"
DATASET_SEP=","
DATASET_FILENAME = 'C:\\ucrise_dataset_project_final\\u_crise_dataset.csv'
df = pd.read_csv(DATASET_FILENAME, sep=DATASET_SEP, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)


# In[71]:


df


# In[72]:


data=df[['claims','status']]


# In[73]:


data['claims'].unique()


# In[74]:


print(f"Total Count = {df.count()}")


# In[75]:


#Lowercase Transformation
dataset = df['claims'].str.lower()


# In[76]:


df


# In[77]:


#Stopwords Removal
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']


# In[78]:


stop_words = set(stopwordlist)
def remove_stop_words(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])
df['claims'] = df['claims'].apply(lambda text: remove_stop_words(text))
df_stopwords = df['claims']
print(f"Stop Words : {df_stopwords}")


# In[79]:


#Punctuations Removal
english_punctuations = string.punctuation
punctuations_list = english_punctuations
def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
df['claims']= df['claims'].apply(lambda x: remove_punctuations(x))
print(df['claims'])


# In[80]:


#Repeating Characters Removal
def remove_repeating_character(text):
    return re.sub(r'(.)1+', r'1', text)
df['claims'] = df['claims'].apply(lambda x: remove_repeating_character(x))
print(df['claims'].head())


# In[81]:


#URL Removal
def remove_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
df['claims'] = df['claims'].apply(lambda x: remove_URLs(x))
print(df['claims'].head())


# In[82]:


#Numbers Removal
def remove_numbers(data):
    return re.sub('[0-9]+', '', data)
df['claims'] = df['claims'].apply(lambda x: remove_numbers(x))
df['claims'].head()


# In[83]:


#Word Tokenization
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'w+')
df['claims'] = df['claims'].apply(tokenizer.tokenize)
df['claims'].head()


# In[84]:


#Stemming
st = nltk.PorterStemmer()
def stem_text(data):
    text = [st.stem(word) for word in data]
    return data
df['claims'] = df['claims'].apply(lambda x: stem_text(x))
df['claims'] = df['claims'].astype('string')

df_stemming = df['claims']

print(df_stemming)


# In[85]:


#TF_IDF

X=data.claims
y=data.status


# In[96]:


# Separating the 95% data for training data and 5% for testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state =26105111)

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
vectoriser.fit(X_train)
print('No. of feature_words: ', len(vectoriser.get_feature_names()))

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)

print(f"X_Train = {X_train}")
print(f"X_test = {X_test}")


# In[93]:


DATASET_ENCODING = "ISO-8859-1"
#  Reads ‘alice.txt’ file
sample = open('C:\\ucrise_dataset_project_final\\u_crise_dataset.csv',encoding=DATASET_ENCODING)
s = sample.read()
# Replaces escape character with space
f = s.replace("\n", " ")
data = []
# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)


# In[94]:


# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100, window = 5)
# Print results
print("Cosine similarity between 'ukraine' " + 
               "and 'war' - CBOW : ",
    model1.wv.similarity('ukraine', 'war'))
     
print("Cosine similarity between 'putin' " +
                 "and 'zelenskyy' - CBOW : ",
      model1.wv.similarity('putin', 'zelenskyy'))
 
print("Cosine similarity between 'ukrainewar' " +
                 "and 'russia' - CBOW : ",
      model1.wv.similarity('ukrainewar', 'russia'))


# In[95]:


# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100,
                                             window = 5, sg = 1)
 
# Print results
print("Cosine similarity between 'ukraine' " +
          "and 'war' - Skip Gram : ",
    model2.wv.similarity('ukraine', 'war'))
     
print("Cosine similarity between 'putin' " +
            "and 'zelenskyy' - Skip Gram : ",
      model2.wv.similarity('putin', 'zelenskyy'))

print("Cosine similarity between 'ukrainewar' " +
                 "and 'russia' - Skip Gram  : ",
      model2.wv.similarity('ukrainewar', 'russia'))


# In[87]:


#Evaluation of Models
def model_Evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Claims','True','False']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)


# In[88]:


'#BernoulliNB Model'
print("############# Bernoulli Model #####################")
BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel)
y_pred1 = BNBmodel.predict(X_test)


# In[89]:


'#Linear SVC model'
print("############# Linear SVC Model #####################")
SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
model_Evaluate(SVCmodel)
y_pred2 = SVCmodel.predict(X_test)


# In[90]:


print("############# Logistic Model #####################")
LRmodel = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)
y_pred3 = LRmodel.predict(X_test)


# In[91]:


print("############# Random Forest Model #####################")
RFmodel = RandomForestClassifier()
RFmodel.fit(X_train, y_train)
model_Evaluate(RFmodel)
y_pred3 = RFmodel.predict(X_test)

