import pickle
import sys

sys.path.append("..")

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import shuffle

from text_processing import TFIDFModel

if __name__ == "__main__":
    
    df = pd.read_csv('../../data/classification/classification.csv')
    df = shuffle(df)
    
    inputs = []
    for input in df['input']:
        inputs.append(input)
    
    classes = []
    for clss in df['class']:
        classes.append(clss)
    
    model = TFIDFModel(model_dir='../../models/classification', transformer_name='intent_trans.pkl', corpus_name='intent.npy')
    
    model.build_tfidf_model(inputs)

    X_train_tf = np.load('../../models/classification/intent.npy', allow_pickle=True)

    clf=SVC(C=7, probability = True).fit(X_train_tf, classes)

    with open('../../models/classification/intentclassifier.pkl', 'wb') as fw:
        pickle.dump(clf, fw)

        
    
        
    