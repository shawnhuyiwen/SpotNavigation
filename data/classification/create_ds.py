import pandas as pd

dest_df = pd.read_csv('data\dest.csv', index_col=False)

commands_df = pd.read_csv('data\commands.csv') 

classification = {"input":[], "class":[]}

for elem in dest_df['sentence']:
    classification['input'].append(elem)
    classification['class'].append('Navi')
    
for elem in commands_df['commands']:
    classification['input'].append(elem)
    classification['class'].append('Comm')
    


pd.DataFrame(classification).to_csv('data/classification/classification.csv')