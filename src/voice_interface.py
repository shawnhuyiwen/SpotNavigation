from speech_recog import Recognizor
from text_processing import TextProcessor, TFIDFModel
import time
import warnings
import pandas as pd
import pickle

warnings.filterwarnings('ignore')

is_activate = False
commands = {
    'sit': 0,
    'down': 0,
    'forward': 1,
    'backwards': 2,
    'left': 3,
    'right': 4
}

model = TFIDFModel(model_dir='./models/classification', transformer_name='intent_trans.pkl', corpus_name='intent.npy')

df = pd.read_csv('./data/classification/classification.csv')
        
dataset = []
for input in df['input']:
    dataset.append(input)

def intent_detection(input):
    
    input_tfidf = model.calculate_input_tfidf(dataset, input)

    classifier_path = './models/classification/intentclassifier.pkl'
    classifier = pickle.load(open(classifier_path, "rb"))

    predicted = classifier.predict(input_tfidf.toarray())
    
    return predicted[0]

while True:
    print("I'm listening")
    rec = Recognizor()
    text_processor = TextProcessor()
    results = rec.initiate_recognizor()

    print(results)

    if(type(results) == str or results == []):
        continue

    text_processor.preprocessing(results)
    
    if not is_activate:
        is_activate = text_processor.check_commands(['spot', 'sport'])[0]
        time.sleep(0.5)
        if is_activate == True:
            print("I'm here!")
    else:
        intent = intent_detection([text_processor.text_list[0]])
        
        if intent == 'Comm':
            detected, command = text_processor.check_commands(['sit', 'down','forward','backwards','left','right'])
            if detected:
                print(commands[command])
        else:  
            dest = text_processor.extract_dest()
            print(dest)
        


    
