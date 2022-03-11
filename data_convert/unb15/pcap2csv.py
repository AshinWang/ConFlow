import glob
import os
from nfstream import NFStreamer
import pandas as pd
import socket
import shutil

import warnings
warnings.filterwarnings("ignore")

import os
os.chdir("/home/ashin/workspace")

class PCAP2CSV_2015():
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
    def get_proto_bysocket(self, proto_name):
        try:
            proto_num = socket.getprotobyname(proto_name)
            
        except:
            proto_num = self.proto_dict[proto_name]
        return proto_num
    
    def get_proto_dict(self, path='datasets/UNSW-NB15/protocol-numbers-1.csv'):
        '''
        protocol-numbers-1.csv from 'https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml'
        and a part of proto dict from 'https://datatracker.ietf.org/doc/html/rfc1340'
        '''
        
        proto_df = pd.read_csv(path)
        proto_df = proto_df.drop(proto_df[proto_df['Decimal']=='144-252'].index)
        proto_df[proto_df['Decimal']=='61']=proto_df[proto_df['Decimal']=='61'].fillna('any')
        proto_df[proto_df['Decimal']=='63']=proto_df[proto_df['Decimal']=='63'].fillna('any')
        proto_df[proto_df['Decimal']=='68']=proto_df[proto_df['Decimal']=='68'].fillna('any')
        proto_df[proto_df['Decimal']=='99']=proto_df[proto_df['Decimal']=='99'].fillna('any')
        proto_df[proto_df['Decimal']=='114']=proto_df[proto_df['Decimal']=='114'].fillna('any')
        proto_df[proto_df['Decimal']=='253']=proto_df[proto_df['Decimal']=='253'].fillna('Use for experimentation and testing')
        proto_df[proto_df['Decimal']=='254']=proto_df[proto_df['Decimal']=='254'].fillna('Use for experimentation and testing')

        for i in range(144, 253):
            proto_df = proto_df.append({'Decimal': i, 'Keyword': 'unas'}, ignore_index=True)

        proto_df['Keyword'] = proto_df['Keyword'].apply(lambda x: str(x).lower())
        proto_df['Keyword'] = proto_df['Keyword'].apply(lambda x: str(x).replace(' (deprecated)', ''))
        proto_df['Decimal'] = proto_df['Decimal'].apply(lambda x: int(x))

        proto_df = proto_df.sort_values('Decimal')
        proto_df = proto_df.reset_index()
        proto_df = proto_df[['Keyword', 'Decimal']]

        proto_dict = proto_df.set_index('Keyword').to_dict('Decimal')['Decimal']

        proto_dict['ipnip'] = 4
        proto_dict['st2'] = 5
        proto_dict['bbn-rcc'] = 10
        proto_dict['nvp'] = 11
        proto_dict['dcn'] = 19
        proto_dict['sep'] = 33
        proto_dict['mhrp'] = 48
        proto_dict['ipv6-no'] = 59
        proto_dict['aes-sp3-d'] = 96
        proto_dict['ipx-n-ip'] = 111
        proto_dict['sccopmce'] = 128
        
        proto_dict['zero'] = -1
        proto_dict['ib'] = -1
        proto_dict['pri-enc'] = -1
        return proto_dict

    def convert_proto_num(self, proto_num):
        if proto_num in [61, 63, 68, 99, 114]:
            proto_num = 114
        elif proto_num == 253 or proto_num == 254:
             proto_num = 254
        elif 144 <= proto_num <= 252:
            proto_num = 252
        else:
            return proto_num
        return proto_num
    

    # 读取标签
    def read_label(self):
        data = pd.read_csv(self.label_file_path)
        
        data_cols = ['timestamp', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Attack category']
        col_list = ['timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'label']
        
        label_df=pd.DataFrame()
        label_df[col_list] = data[data_cols]       
        self.proto_dict = self.get_proto_dict()
        label_df['protocol'] = label_df['protocol'].apply(lambda x: self.get_proto_bysocket(x))
        label_df['protocol'] = label_df['protocol'].astype('int')
        return label_df
    
    def convert_time(self, time):
        new_time = time / 1000 
        return int(new_time)
    
   
    # 标签连接
    def add_label(self, label_df, pcap_file):
        nfs_data = self.pcap_to_df(pcap_file)
        
        nfs_data['protocol'] = nfs_data['protocol'].apply(lambda x: self.convert_proto_num(x))
        nfs_data['timestamp'] = nfs_data['bidirectional_first_seen_ms'].apply(lambda x: self.convert_time(x))
        
        mer_key = ['timestamp', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol']
        
        
        labeled_data = pd.merge(nfs_data, label_df, on=mer_key, how='left')
        labeled_data['label'] = labeled_data['label'].fillna('Normal')

        labeled_data.drop_duplicates(subset=['id'], keep=False, inplace=True) 
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
                    # timestamp
                    'timestamp',
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
           

if __name__ == '__main__':

    label_file_path = 'datasets/UNSW-NB15/label.csv'

    pcap_path_1 = 'datasets/UNSW-NB15/UNSW-NB15-PCAP/2022-1-2015/'
    pcap_path_2 = 'datasets/UNSW-NB15/UNSW-NB15-PCAP/2017-2-2015/'

    saved_path_1 = 'datasets/UNSW-NB15/UNSW-NB15-PCAP2CSV/2022-1-2015/'
    saved_path_2 = 'datasets/UNSW-NB15/UNSW-NB15-PCAP2CSV/2017-2-2015/'

    pcap2csv_1 = PCAP2CSV_2015(pcap_path_1, label_file_path, saved_path_1)
    pcap2csv_2 = PCAP2CSV_2015(pcap_path_2, label_file_path, saved_path_2)
    
    pcap2csv_2.match()
    pcap2csv_1.match()


    filenames_1 = glob.glob(saved_path_1 + "*.csv")
    filenames_2 = glob.glob(saved_path_2 + "*.csv")

    dfs = []
    for file in filenames_1:
        dfs.append(pd.read_csv(file))
    for file in filenames_2:
        dfs.append(pd.read_csv(file))
    data_df = pd.concat(dfs)

    print(data_df['label'].value_counts())
    data_df.to_csv('datasets/UNSW-NB15/merge_unsw15.csv', index=False, encoding='utf-8')