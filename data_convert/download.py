import os
import threading
import requests
from tqdm import tqdm


def download_2012(i, saved_path):
    if not os.path.exists(saved_path + 'labeled_flows_xml.zip'):
        xml_label = 'http://205.174.165.80/CICDataset/ISCX-IDS-2012/Dataset/labeled_flows_xml.zip -P {saved_path}'
        os.system(xml_label)
    cmd = 'wget http://205.174.165.80/CICDataset/ISCX-IDS-2012/Dataset/testbed-1{}jun.pcap -P {saved_path}'.format(i, saved_path)
    os.system(cmd)


def download_2015(date, start, end, saved_path):
    '''
    date: str('2022-1-2015', '2017-2-2015')
    start: int=1
    end: int=54/28
    '''

    for i in tqdm(range(start, end)):
        url = 'https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download?path=%2FUNSW-NB15%20-%20pcap%20files%2Fpcaps%{}&files={}.pcap&downloadStartSecret=4oppygn2jhg'.format(date, i)
        r = requests.get(url, stream=True)

        f = open(saved_path + "{}/{}.pcap".format(date, i), "wb")
        try:
            for chunk in r.iter_content(chunk_size=1024*32):
                if chunk:
                    f.write(chunk)
        except:
            continue
        else:
            print('    ')
            print('{}/{}.pcap Done!'.format(date, i))


def download_2017(saved_path):
    '''
    http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/
    '''
    cmd = 'wget -c -r http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/ -P {}'.format(saved_path)
    os.system(cmd)


def download_2019(saved_path):
    '''
    http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/
    '''
    cmd = 'wget -c -r http://205.174.165.80/CICDataset/CICDDoS2019/Dataset/ -P {}'.format(saved_path)
    os.system(cmd)


if __name__ == '__main__':
    print('############## Downloading ISCX-IDS-2012 ##############')
    saved_path = 'datasets/ISCX-IDS-2012/'
    for i in range(1, 8):
        t = threading.Thread(target=download_2012, args=(i, saved_path))
        t.start()
    print(threading.active_count())

    