import xml.etree.ElementTree as ET
from lxml import etree
import pandas as pd
import glob
import re
import time
import socket
import os
os.environ['TZ'] = 'America/Araguaina'
os.chdir("/home/ashin/workspace")

def xml2df(xml_path):
    '''
    1276656842113: TimeStamp
    2010-06-16 10:54:02 UTC+8
    2010-06-16 02:54:02 UTC+0
    2010-06-15 23:54:02 UTC-3
    '''
    xml_data = open(xml_path, 'r').read() 
    xml_data = re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f]+",u"", xml_data)
    parser = etree.XMLParser(ns_clean=True, recover = True)
    root = ET.fromstring(xml_data, parser=parser)
   
    data = []
    for i, child in enumerate(root):
        data.append([subchild.text for subchild in child])
    for child in root:
        cols = [subchild.tag for subchild in child]
        break
    df = pd.DataFrame(data) 
    df.columns = cols  
    df = df[['source', 'protocolName', 'sourcePort', 'destination', 'destinationPort', 'startDateTime', 'stopDateTime', 'Tag']]
    return df

def convert_strtime(strtime):
    strtime=time.strptime(strtime,'%Y-%m-%dT%H:%M:%S')
    time_stamp=time.mktime(strtime)
    return time_stamp

def get_proto_bysocket(proto_name):
    proto_num = socket.getprotobyname(proto_name)     
    return proto_num

if __name__ == '__main__':
    filenames = glob.glob('datasets/ISCX-IDS2012/xml_label/' + "*.xml")
    dfs = []
    for file in filenames:
        print('\033[1;33mStart parsing XML: {} \033[0m'.format(file))
        dfs.append(xml2df(file))
        print('\033[1;32mDone! Convert DataFrame successfully! \033[0m')
        print()
    data_df = pd.concat(dfs)
    data_df['protocolName'] = data_df['protocolName'].apply(lambda x: get_proto_bysocket(str(x).replace('_ip', '').replace('ipv6', ''))) 
    data_df['startDateTime'] = data_df['startDateTime'].apply(lambda x: convert_strtime(x))    
    data_df['stopDateTime'] = data_df['stopDateTime'].apply(lambda x: convert_strtime(x))   

    data_df.to_csv('datasets/ISCX-IDS2012/label.csv', index=False)