import en_core_web_trf
import warnings

import pandas as pd

df = pd.read_csv('./data/dest.csv', sep=',', index_col = False)

print(df)

dict1 = {'sentence': df['sentence'], 'start':[], 'end':[], 'entity':[]}

for row in df.itertuples():
    sentence = getattr(row, 'sentence')
    entity = getattr(row, 'entity')
    
    start = sentence.find(entity)
    end = start + len(entity)
    
    dict1['start'].append(start)
    dict1['end'].append(end)
    dict1['entity'].append("DEST")
    

loc = pd.DataFrame.from_dict(dict1)

training_data = []
validatin_data = [
    ['where is room 2', {'entities': [(9, 15, 'DEST')]}],
    ['can you take me to the gym', {'entities': [(23, 26, 'DEST')]}],
    ['do you know where is the auditorium', {'entities': [(25, 35, 'DEST')]}],
    ['can you show me the way to room 45', {'entities': [(27, 34, 'DEST')]}],
    ['guide me to room c 4', {'entities': [(12, 20, 'DEST')]}],
]

for row in loc.itertuples():
    sentence = getattr(row, 'sentence')
    entity = getattr(row, 'entity')
    start = getattr(row, 'start')
    end = getattr(row, 'end')
    
    temp = []
    
    temp.append(sentence)
    temp.append({"entities": [(start, end, entity)]})
    
    training_data.append(temp)


import spacy
from spacy.tokens import DocBin

nlp = spacy.load('en_core_web_lg')
# the DocBin will store the example documents
db = DocBin()
for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        if span is None:
            msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
            warnings.warn(msg)
        else:
            ents.append(span)
    doc.ents = ents
    db.add(doc)
db.to_disk("./data/train.spacy")

for text, annotations in validatin_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        if span is None:
            msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
            warnings.warn(msg)
        else:
            ents.append(span)
    doc.ents = ents
    db.add(doc)
db.to_disk("./data/dev.spacy")