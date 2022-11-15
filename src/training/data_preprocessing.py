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

for row in loc.itertuples():
    sentence = getattr(row, 'sentence')
    entity = getattr(row, 'entity')
    start = getattr(row, 'start')
    end = getattr(row, 'end')
    
    temp = []
    
    temp.append(sentence)
    temp.append({"entities": [(start, end, entity)]})
    
    training_data.append(temp)

print(training_data)
    

    


    
    