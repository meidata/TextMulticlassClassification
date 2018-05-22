
# coding: utf-8

# Cette multi-label classification est faite par le modèle de SVM (linearSVC) avec la 5-folds cross-validation stratifiée en obtenant le taux de précision moyenne de **97.245%** sur l'ensemble de l'échantillon d'apprentissage.

# * [Text Prepocessing](#Text-Prepocessing)
# * [Features Engineering](#Features-Engineering)
# * [Model Creation](#Modeling)
# * [Model Accuracy](#Model-average-accuracy)

# In[1]:

import gc
import re
import regex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from scipy.sparse import hstack

pd.set_option('display.max_colwidth', -1)
pd.option_context('display.max_rows', 20000)


# In[2]:

train = pd.read_csv('train.csv',encoding='latin1')
test = pd.read_csv('test.csv',encoding='latin1')
category_index = train.groupby('category_id')['category'].unique().reset_index()
del train['category']
gc.collect()


# ### Text Prepocessing

# In[3]:

def getUpper(row):
    row['entityUP'] = ' '.join(re.findall(r'[A-Z]{2,}', row['title']))
    return row

train = train.apply(getUpper,axis=1)
test = test.apply(getUpper,axis=1)


# In[4]:

# prepocessing the text

class BaseReplacer:
    def __init__(self, pattern_replace_pair_list=[]):
        self.pattern_replace_pair_list = pattern_replace_pair_list
    def transform(self, text):
        for pattern, replace in self.pattern_replace_pair_list:
            try:
                text = regex.sub(pattern, replace, text)
            except:
                pass
        return regex.sub(r"\s+", " ", text).strip()
   
        
class lowerCaseConverter(BaseReplacer):
    """
    Traditional -> traditional
    """
    def transform(self, text):
        return text.lower()
        

class letterLetterSplitter(BaseReplacer):
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"([a-zA-Z]+)[/\-|]([a-zA-Z]+)", r"\1 \2"),]        
        
class numberRegularizer(BaseReplacer):
    '''
    4.7 ---> 4_7
    4,7 ---> 4_7
    750/000 ---> 750_000
    '''
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"([\d]+)[\.]([\d]+)", r"\1_\2"),
            (r"([\d]+)[\,]([\d]+)", r"\1_\2"),
            (r"([\d]+)[\/]([\d]+)", r"\1_\2"),
        ]       
        
class inchCombiner(BaseReplacer):
    '''
    4,7 '' ---> 4,7 inch
    4,7'' ---> 4,7 inch
    4,7" ---> 4,7 inch
    
    '''
    def __init__(self):
        self.pattern_replace_pair_list = [
         (r"([\d]+[\_][\d])(.)?('')", r'\1 inch'),
         (r"([\d]+[\_][\d])(.)?(\")", r'\1 inch')            
        ]
        
class abbRemover(BaseReplacer):
    '''
    d'ecran ---> ecran
    '''
    def __init__(self):
        self.pattern_replace_pair_list = [
         (r"([dl])(')(\w+)", r'\3')       
        ]
        
class punctunationCleaner(BaseReplacer):
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"-", r" "),
            (r"\.", r" "),
            (r"\|", r" "),
            (r"\:", r" "),
            (r'\"', r" "),
            (r'\)', r" "),
            (r'\(', r" "),
            (r'\+', r" "),
            (r'\!', r" "),
            (r'\*', r" "),
            (r'\#', r" "),
            (r'\/', r" "),
            (r'\®', r" "),
            (r'\,', r" "),
            (r'\&', r" "),
        ]
        
class sentenceRemover(BaseReplacer):
    '''
    Voir la presentation---> ''
    '''
    def __init__(self):
        self.pattern_replace_pair_list = [
         (r"(voir la pr)(\w+)", r'')       
        ]
        
class numberDigitMapper(BaseReplacer):
    """
    one -> 1
    two -> 2
    """
    def __init__(self):
        numbers = [
            'zero', 'un' ,  'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 
            'huit', 'neuf', 'dix'
        ]
        digits = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        ]
        self.pattern_replace_pair_list = [
            (r"(?<=\W|^)%s(?=\W|$)"%n, str(d)) for n,d in zip(digits,numbers)
        ]

class stopwordRemover(BaseReplacer):
    def __init__(self):
        stop = set(stopwords.words('french'))
        self.pattern_replace_pair_list = [
        (r"\b(%s)+\b"%w,'') for w in stop 
    ]

class lenFilter(BaseReplacer):
    def transform(self, text):
        return ' '.join([w for w in text.split(' ') if len(w)>1])


# In[5]:

processors=[
    lowerCaseConverter(),
    letterLetterSplitter(),
    numberRegularizer(),
    inchCombiner(),
    abbRemover(),
    punctunationCleaner(),
    sentenceRemover(),
    numberDigitMapper(),
    stopwordRemover(),
    lenFilter()
]

def prepossessing(row):
    for processor in processors:
        row = processor.transform(row)
    return row

train['desc_prepossed'] = train['description'].apply(prepossessing)
train['title_prepossed'] = train['title'].apply(prepossessing)
test['desc_prepossed'] = test['description'].apply(prepossessing)
test['title_prepossed'] = test['title'].apply(prepossessing)
train['entityUP'] = train['entityUP'].apply(lambda x:lowerCaseConverter().transform(x))
test['entityUP'] = test['entityUP'].apply(lambda x:lowerCaseConverter().transform(x))


# ### Features Engineering 
# #### Text Vectorization

# In[6]:

# #############################################################################
# vectorizing the description

train_text = train['desc_prepossed']
test_text = test['desc_prepossed']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3),
    min_df = 2,
    max_features=100000)

word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)


# In[7]:

# vectorizing the title

train_text = train['title_prepossed']
test_text = test['title_prepossed']
all_text = pd.concat([train_text, test_text])

word_vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        max_features=50000)

word_vectorizer.fit(all_text)
train_title_features = word_vectorizer.transform(train_text)
test_title_features = word_vectorizer.transform(test_text)


# In[8]:

# vectorizing the feature entityUPPER

train_text = train['entityUP']
test_text = test['entityUP']
all_text = pd.concat([train_text, test_text])

word_vectorizer = CountVectorizer(
        #ngram_range=(1, 2),
        max_features=10000)
       #,min_df=2)

word_vectorizer.fit(all_text)
train_entity_features = word_vectorizer.transform(train_text)
test_entity_features = word_vectorizer.transform(test_text)


# In[9]:

# vectorizing the description by level of char

train_text = train['desc_prepossed']
test_text = test['desc_prepossed']
all_text = pd.concat([train_text, test_text])

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(4, 6),
    min_df=2,
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)


# In[10]:

### Assembling of the features

train_features = hstack([train_title_features, train_word_features,train_entity_features,train_char_features])
test_features = hstack([test_title_features, test_word_features,test_entity_features,test_char_features ])


# ### Modeling 
# #### SVM model with 5-fold CV

# In[11]:

from sklearn.model_selection import StratifiedKFold
X = train_features.tocsc()
y = train.category_id.values
skf = StratifiedKFold(n_splits=5, random_state=123)
skf.get_n_splits(X, y)
print(skf)


# In[12]:

test_accuracy_svm = []
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    svm = OneVsRestClassifier(LinearSVC(), n_jobs=2)
    svm.fit(X_train, y_train)
    prediction = svm.predict(X_test)
    ac_score = accuracy_score(y_test, prediction)
    test_accuracy_svm.append(ac_score)
    print('Test accuracy is {}'.format(accuracy_score(y_test, prediction)))


# #### Model average accuracy

# In[13]:

np.mean(test_accuracy_svm)


# In[14]:

test['predicted_category_id'] =  svm.predict(test_features)
test[['id','predicted_category_id']].to_csv('predictions.csv')

