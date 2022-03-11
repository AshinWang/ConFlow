import time
import glob
import os
from datetime import datetime
from datetime import datetime, timedelta, timezone
from nfstream import NFStreamer
import pandas as pd
import shutil

class PCAP2CSV_2017():
    '''
    pcap 转换为 df, tuple(src_ip, src_port, dst_ip, dst_port, protocol)
    https://github.com/ahlashkari/CICFlowMeter/blob/master/src/main/java/cic/cs/unb/ca/ifm/CICFlowMeter.java# 
    '''
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
        print('-------pcap_to_df-------')
        print(df.shape[0])
        return df

    # 读取标签
    def read_label(self, pcap_file):
        
        file_name_map = {"Friday-WorkingHours.pcap": ["Friday-WorkingHours-Morning.pcap_ISCX.csv",
                                                   "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", 
                                                   "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"], 
                      "Thursday-WorkingHours.pcap": ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                                                     "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"],
                      "Wednesday-WorkingHours.pcap": ["Wednesday-workingHours.pcap_ISCX.csv"],
                      "Tuesday-WorkingHours.pcap": ["Tuesday-WorkingHours.pcap_ISCX.csv"], #brute
                      "Monday-WorkingHours.pcap": ["Monday-WorkingHours.pcap_ISCX.csv"]} # benign

        pcap_file = pcap_file.split('/')[-1]
        label_file_list = file_name_map[pcap_file] 

        dfs = []
        for label_file in label_file_list:
            csv_file_path = os.path.join(self.label_file_path, label_file)
            dfs.append(pd.read_csv(csv_file_path, keep_default_na=True, encoding='cp1252'))
        label_df = pd.concat(dfs)
        
        sel_col = [' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Protocol', ' Timestamp', ' Label']
        new_col = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'time', 'label']
        label_df = label_df[sel_col]
        label_df.columns = new_col

        # label conuts
        print()
        print('-------official_label-------')
        print(label_df['label'].value_counts())
        print(label_df.shape[0])
        return label_df
    
    # 时间戳转换
    def convert_time_fun(self, time_stamp):
        # 时区 UTC -3
        time_stamp = float(time_stamp)/1000.0
        td = timedelta(hours = -3)
        c_timestamp = datetime.fromtimestamp(time_stamp, tz=timezone(td))

        time_arr = time.strptime((str(c_timestamp)[:16]), '%Y-%m-%d %H:%M')
        convert_time = time.strftime("%d-%m-%Y %H:%M", time_arr)

        # 7/7/2017 8:59
        day_ = str(c_timestamp.day)
        month_ = str(c_timestamp.month)
        year_ = str(c_timestamp.year)

        if c_timestamp.hour > 12:
            hour_ = str(c_timestamp.hour - 12)
        else:
            hour_ = str(c_timestamp.hour)

        if len(str(c_timestamp.minute)) == 1:
            minute_ = '0' + str(c_timestamp.minute)
        else:
            minute_ = str(c_timestamp.minute)

        convert_time = day_ + '/' + month_ + '/' + year_ + ' ' + hour_ + ':' + minute_
        return convert_time

    # 标签连接
    def add_label(self, pcap_file):
        print(pcap_file)
        nfs_data = self.pcap_to_df(pcap_file)
        label_df = self.read_label(pcap_file)
        
        nfs_data['time'] = nfs_data['bidirectional_first_seen_ms'].apply(lambda x: self.convert_time_fun(x))

        if pcap_file == 'datasets/CIC-IDS2017/PCAPs/Monday-WorkingHours.pcap':
            labeled_data = nfs_data
            labeled_data['label'] = 'BENIGN'
        else:
            mer_key = ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol', 'time']
            labeled_data = pd.merge(nfs_data, label_df, how='left', on=mer_key)
            labeled_data.drop_duplicates(subset=['id'], keep='first', inplace=True)
            labeled_data.dropna(subset=['label'], inplace=True)

        drop_col = ['id', 
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
                    'bidirectional_first_seen_ms', 
                    'bidirectional_last_seen_ms', 
                    'src2dst_first_seen_ms', 
                    'src2dst_last_seen_ms', 
                    'dst2src_first_seen_ms', 
                    'dst2src_last_seen_ms',
                    'time',

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
        pcap_file = pcap_file.split('/')[-1]
        saved_csv_path = os.path.join(self.saved_path, str(pcap_file + '.csv'))                   
        labeled_data.to_csv(saved_csv_path, index=False)
            
    
    def match(self):
        self.mk_dir()
        filenames = glob.glob(self.pcap_file_path + "*.pcap")
       
        for file in filenames:
            print('\033[1;33mStart parsing package: {} \033[0m'.format(file))
            self.add_label(file)
            print('\033[1;32mDone! Convert CSV successfully! \033[0m')
            print()
            print()


if __name__ == '__main__':
    os.chdir("/home/ashin/workspace/")

    pcap_dir_path = 'datasets/CIC-IDS2017/PCAPs/'
    label_file_path = 'datasets/CIC-IDS2017/TrafficLabelling/'
    saved_path = 'datasets/CIC-IDS2017/PCAP2CSV/'
    pcap2csv = PCAP2CSV_2017(pcap_dir_path, label_file_path, saved_path)
    pcap2csv.match()

    filenames = glob.glob(saved_path + "*.csv")
    dfs = []
    for file in filenames:
        dfs.append(pd.read_csv(file))
    data_df = pd.concat(dfs)

    label_code = {'BENIGN': 'BENIGN',
                'DoS Hulk': 'DoS',
                'PortScan': 'PortScan',
                'DDoS': 'DDoS',
                'DoS GoldenEye': 'DoS',
                'FTP-Patator': 'Patator',
                'SSH-Patator': 'Patator',
                'DoS slowloris': 'DoS',
                'DoS Slowhttptest': 'DoS',
                'Bot': 'Bot',
                'Web Attack – Brute Force': 'Web Attack',
                'Web Attack – XSS': 'Web Attack',
                'Web Attack – Sql Injection': 'Web Attack',
                'Infiltration': 'Infiltration',
                'Heartbleed': 'DoS',
                }

    data_df['label'] = [label_code[item] for item in data_df['label']]
    print(data_df['label'].value_counts())
    data_df.to_csv('datasets/CIC-IDS2017/merge_ids17.csv', index=False, encoding='utf-8')