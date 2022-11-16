import nltk
import spacy
import pickle

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class TextProcessor:
    def __init__(self):
        self.nlp_model = spacy.load("./models/model-best")
        self.text_list = []
        self.tok_list = []
    
    def preprocessing(self, results):
        for result in results['alternative']:
            self.text_list.append(result['transcript'])

        tokenizer = nltk.RegexpTokenizer(r"\w+") 

        for result in self.text_list:
            self.tok_list.append(tokenizer.tokenize(result.lower()))
        
        # print(self.tok_list)

        return self.tok_list

    def check_commands(self, keywords):
        for keyword in keywords:
            if keyword in self.tok_list[0]:
                return True, keyword
            for command in self.tok_list:
                if keyword in command:
                    return True, keyword
        
        return False, ''
    
    def extract_dest(self):
        ent_list = []
        for text in self.text_list:
            doc = self.nlp_model(text)
            for ent in doc.ents:
                if ent.label_ == "DEST":
                    ent_list.append(str(ent.text))
        
        return ent_list
            

class TFIDFModel:
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
        
        
