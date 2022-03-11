import pandas as pd
from tqdm.autonotebook import tqdm
import os
os.chdir("/home/ashin/workspace")

label_df = pd.read_csv('datasets/UNSW-NB15/UNSW-NB15 - CSV Files/NUSW-NB15_GT.csv')
label_df['Attack category'] = label_df['Attack category'].apply(lambda x: x.replace(' ', ''))
label_df['Attack category'] = label_df['Attack category'].apply(lambda x: x.replace('Backdoors', 'Backdoor'))
label_df['Delta time'] = label_df['Last time'] - label_df['Start time'] 
label_df = label_df[['Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Start time', 'Last time', 'Attack category', 'Delta time']]

new_label_df = pd.DataFrame()
for index, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
    if row['Delta time'] == 0:
        new_label_df = new_label_df.append({
                         'timestamp': int(row['Start time']),
                         'Source IP': row['Source IP'], 
                         'Source Port': row['Source Port'], 
                         'Destination IP': row['Destination IP'], 
                         'Destination Port': row['Destination Port'], 
                         'Protocol': row['Protocol'],
                         'Attack category': row['Attack category']} , 
                        ignore_index=True)
    else:
        dict_list = []
        for i in range(0, row['Delta time'] + 1):
            time = row['Start time'] + i 
            dict_list.append({
                             'timestamp': int(time),
                             'Source IP': row['Source IP'], 
                             'Source Port': row['Source Port'], 
                             'Destination IP': row['Destination IP'], 
                             'Destination Port': row['Destination Port'], 
                             'Protocol': row['Protocol'],
                             'Attack category': row['Attack category']}
            )
        new_label_df = new_label_df.append(dict_list)
print(new_label_df['Attack category'].value_counts())
new_label_df.to_csv('datasets/UNSW-NB15/label.csv')
print('\033[1;32mDone! Convert Ground Truth successfully! \033[0m')