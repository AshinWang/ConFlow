import glob
import os
from nfstream import NFStreamer
import pandas as pd
import socket
import shutil

import warnings
warnings.filterwarnings("ignore")
os.chdir("/home/ashin/workspace")

class PCAP2CSV_2012():
    def __init__(self, pcap_dir_path, label_file_path, saved_path):
        self.pcap_file_path = pcap_dir_path
        self.label_file_path = label_file_path
        self.saved_path = saved_path
    
    def mk_dir(self):
        if not os.path.exists(self.saved_path):
            os.makedirs(self.saved_path)
        else:
            shutil.rmtree(self.saved_path)
            os.makedirs(self.saved_path)

    def pcap_to_df(self, pcap_file):
        my_streamer = NFStreamer(source=pcap_file,
                                 decode_tunnels=True,
                                 bpf_filter=None,
                                 promiscuous_mode=True,
                                 snapshot_length=1536, 
                                 idle_timeout=15, 
                                 active_timeout=1800, 
                                 accounting_mode=0,
                                 udps=None,
                                 n_dissections=20,
                                 statistical_analysis=True,
                                 splt_analysis=0,
                                 n_meters=0,
                                 performance_report=0)
        
        df = my_streamer.to_pandas()
        df = df.dropna(subset=df.columns[1:], how='all')
        df[['src_port', 
            'dst_port', 
            'protocol', 
            'bidirectional_first_seen_ms', 
            'bidirectional_last_seen_ms']] = df[['src_port', 
                                                'dst_port', 
                                                'protocol', 
                                                'bidirectional_first_seen_ms', 
                                                'bidirectional_last_seen_ms']].fillna(value=0)

        df[['src_port', 'dst_port', 'protocol']] = df[['src_port', 'dst_port', 'protocol']].astype('int')
        print('-------pcap_to_df-------')
        print(df.shape[0])
        return df

    # 读取标签
    def read_label(self):
        data = pd.read_csv(self.label_file_path)
        data_cols = ['source', 'protocolName', 'sourcePort', 'destination', 'destinationPort', 'startDateTime', 'stopDateTime', 'Tag']
        new_cols = ['src_ip', 'protocol', 'src_port', 'dst_ip', 'dst_port','startDateTime', 'stopDateTime', 'label']
        label_df = pd.DataFrame()
        label_df[new_cols] = data[data_cols]
        return label_df
    
    def convert_time(self, time):
        new_time = time / 1000 
        return int(new_time)
    
   
    # 标签连接
    def add_label(self, label_df, pcap_file):
        nfs_data = self.pcap_to_df(pcap_file)
        nfs_data['bidirectional_first_seen_ms'] = nfs_data['bidirectional_first_seen_ms'].apply(lambda x: self.convert_time(x))
        nfs_data['bidirectional_last_seen_ms'] = nfs_data['bidirectional_last_seen_ms'].apply(lambda x: self.convert_time(x))

        mer_key = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol']
        labeled_data = pd.merge(nfs_data, label_df, on=mer_key, how='left')

        labeled_data['label_check_1'] = (labeled_data['bidirectional_first_seen_ms'] - labeled_data['startDateTime']).apply(lambda x: True if abs(x)<15000 else False)
        labeled_data['label_check_2'] = (labeled_data['bidirectional_last_seen_ms'] - labeled_data['stopDateTime']).apply(lambda x: True if abs(x)<15000 else False)
        labeled_data['label_check'] = labeled_data['label_check_1'] | labeled_data['label_check_2']
        labeled_data = labeled_data[labeled_data['label_check'] == True]        
        
        
        if pcap_file.split('-')[-2] == '11jun':
            labeled_data['label'] = 'Normal'
        elif pcap_file.split('-')[-2] == '12jun':
            labeled_data['label'] = labeled_data['label'].apply(lambda x: 'Normal' if x == 'Normal' else 'Infiltration')
        elif pcap_file.split('-')[-2] == '13jun':
            labeled_data['label'] = labeled_data['label'].apply(lambda x: 'Normal' if x == 'Normal' else 'Infiltration')
        elif pcap_file.split('-')[-2] == '14jun':
            labeled_data['label'] = labeled_data['label'].apply(lambda x: 'Normal' if x == 'Normal' else 'DoS')
        elif pcap_file.split('-')[-2] == '15jun':
            labeled_data['label'] = labeled_data['label'].apply(lambda x: 'Normal' if x == 'Normal' else 'Bot')
        elif pcap_file.split('-')[-2] == '16jun':
            labeled_data['label'] = 'Normal'
        elif pcap_file.split('-')[-2] == '17jun':
            labeled_data['label'] = labeled_data['label'].apply(lambda x: 'Normal' if x == 'Normal' else 'Brute force')
        
            
        labeled_data.drop_duplicates(subset=['id'], keep='first', inplace=True) 
    
        drop_col = [# id
                    'id', 
                    'expiration_id', 
                    'ip_version', 
                    'vlan_id', 
                    'tunnel_id',
                    'src_ip', 
                    'src_mac', 
                    'src_oui', 
                    'dst_ip', 
                    'dst_mac', 
                    'dst_oui', 

                    'startDateTime',
                    'stopDateTime',
                    'label_check_1',
                    'label_check_2',
                    'label_check',
                    
                    'bidirectional_first_seen_ms', 
                    'bidirectional_last_seen_ms', 

                    'src2dst_first_seen_ms', 
                    'src2dst_last_seen_ms', 
                    'dst2src_first_seen_ms', 
                    'dst2src_last_seen_ms',
                    
                    # L7 Features: Most of them are missing. 
                    'requested_server_name', 
                    'client_fingerprint',
                    'server_fingerprint', 
                    'user_agent', 
                    'content_type']

        labeled_data.drop(columns=drop_col, inplace=True)
        # label conuts
        
        print('-------merge_label-------')
        print(labeled_data['label'].value_counts())
        print(labeled_data.shape[0])
      
        saved_csv_path = os.path.join(self.saved_path, str(pcap_file.split('/')[-1] + '.csv'))                   
        labeled_data.to_csv(saved_csv_path, index=False)
            
    def match(self):
        self.mk_dir()
        filenames = glob.glob(self.pcap_file_path + "*.pcap")
        label_df = self.read_label()
        for file in filenames:
            print('\033[1;33mStart parsing package: {} \033[0m'.format(file))
            self.add_label(label_df, file)
            print('\033[1;32mDone! Convert CSV successfully! \033[0m')
            print()
            print()

if __name__ == '__main__':
    label_file_path = 'datasets/ISCX-IDS2012/label.csv'
    pcap_path = 'datasets/ISCX-IDS2012/splitpcaps/'
    saved_path = 'datasets/ISCX-IDS2012/PCAP2CSV/'

    pcap2csv = PCAP2CSV_2012(pcap_path, label_file_path, saved_path)
    pcap2csv.match()

    filenames = glob.glob(saved_path + "*.csv")
    dfs = []
    for file in filenames:
        dfs.append(pd.read_csv(file))
    data_df = pd.concat(dfs)

    print(data_df['label'].value_counts())
    data_df.to_csv('datasets/ISCX-IDS2012/merge_iscx12.csv', index=False, encoding='utf-8')
    print(data_df['label'].value_counts().keys())