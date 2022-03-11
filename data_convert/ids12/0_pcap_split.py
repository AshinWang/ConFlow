import os
import glob
import subprocess


os.chdir("/home/ashin/workspace")
def pcap_splitter(pcap_file, output_dir, size):
    cmd = 'PcapSplitter -f {} -o {} -m file-size -p {}'.format(pcap_file, output_dir, size)  
    print(cmd)
    

pcap_file_path = 'datasets/ISCX-IDS2012/PCAP/'
output_dir = 'datasets/ISCX-IDS2012/splitpcaps/'
filenames = glob.glob(pcap_file_path + "*.pcap")

for file in filenames:
    print('\033[1;33mStart spliting package: {} \033[0m'.format(file))
    ret = pcap_splitter(file, output_dir, size=1024*1024*1024*4)
    print(ret)
    print('\033[1;32mDone! Split packet successfully! \033[0m')
    print()