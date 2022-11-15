import os
import pickle
import sys
sys.path.append("..")

from src.classification import Models
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import confusion_matrix

test_data = {
    'input': ['forward',
              'sit down',
              'move backwards',
              'come forward',
              'do you know how to turn left', 
              'do you know where is the lab', 
              'show me the way to the lab',
              'where is the cobot makerspace',
              'can you take me to room a 23',
              'do you know the way to lab 23'],
    'class': ['Comm',
              'Comm',
              'Comm',
              'Comm',
              'Comm',
              'Navi',
              'Navi',
              'Navi',
              'Navi',
              'Navi']
}

model = Models(model_dir='../models/classification', transformer_name='intent_trans.pkl', corpus_name='intent.npy')

df = pd.read_csv('../data/classification/classification.csv')
df = shuffle(df)
    
inputs = []
for input in df['input']:
    inputs.append(input)

input_tfidf = model.calculate_input_tfidf(inputs, test_data['input'])

classifier_path = '../models/classification/intentclassifier.pkl'
classifier = pickle.load(open(classifier_path, "rb"))

predicted = classifier.predict(input_tfidf.toarray())

prob = classifier.predict_proba(input_tfidf.toarray())

print(prob)

# print(confusion_matrix(test_data['class'], predicted))