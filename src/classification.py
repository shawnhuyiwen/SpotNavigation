import os
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import log10
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from nltk.stem.snowball import PorterStemmer
from scipy import spatial
from sklearn.svm import SVC
from sklearn.utils import shuffle


class Models:
    def __init__(self, model_dir, transformer_name, corpus_name):
        self.sb_stemmer = SnowballStemmer('english')
        self.analyzer = CountVectorizer().build_analyzer()
        self.stem_vectorizer = CountVectorizer(analyzer=self.stemmed_words, stop_words=stopwords.words('english'), ngram_range=(1 ,3))
        self.model_dir = model_dir
        self.transformer_path = model_dir + '/' + transformer_name
        self.corpus_path = model_dir + '/' + corpus_name     
           
    def stemmed_words(self, doc):
        return(self.sb_stemmer.stem(w) for w in self.analyzer(doc))
        
    def build_tfidf_model(self, data):
        questions_count = self.stem_vectorizer.fit_transform(data)

        tfidf_transformer = TfidfTransformer(use_idf=True,sublinear_tf=True).fit(questions_count)
        corpus_tfidf = tfidf_transformer.transform(questions_count).toarray()
        
        np.save(self.corpus_path, corpus_tfidf)
        
        tfidftransformer_path = self.transformer_path
        with open(tfidftransformer_path, 'wb') as fw:
            pickle.dump(tfidf_transformer, fw)
    
    def calculate_input_tfidf(self, data, input):
        
        self.stem_vectorizer.fit_transform(data)
        # tfidf_transformer = TfidfTransformer(use_idf=True,sublinear_tf=True).fit(questions_count)
        
        tfidftransformer_path = self.transformer_path
        tfidf_transformer = pickle.load(open(tfidftransformer_path, "rb"))
        
        input_count = self.stem_vectorizer.transform(input)
        input_tfidf = tfidf_transformer.transform(input_count)
        
        return input_tfidf


if __name__ == "__main__":
    
    df = pd.read_csv('../data/classification/classification.csv')
    df = shuffle(df)
    
    inputs = []
    for input in df['input']:
        inputs.append(input)
    
    classes = []
    for clss in df['class']:
        classes.append(clss)
    
    model = Models(model_dir='../models/classification', transformer_name='intent_trans.pkl', corpus_name='intent.npy')
    
    model.build_tfidf_model(inputs)

    X_train_tf = np.load('../models/classification/intent.npy', allow_pickle=True)

    clf=SVC(C=7, probability = True).fit(X_train_tf, classes)

    with open('../models/classification/intentclassifier.pkl', 'wb') as fw:
        pickle.dump(clf, fw)

        
    
        
    