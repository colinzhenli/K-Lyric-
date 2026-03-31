import pandas as pd
import os
#将txt文件转成csv文件

data_path = '../ChineseLyrics'


def get_csv(file):
    df = pd.read_csv(file,delimiter=";",names=['src','tgt'])  #‘;’是分隔符
    # df.columns = ['id','file','text']
    # encoding='utf_8_sig'解决存储csv的乱码问题
    df.to_csv(f"{file}.csv", encoding='utf_8_sig', index=False)
    return 


def main():
    for file in ['train.txt','valid.txt', 'test.txt']:
        file_path = os.path.join(data_path, file)
        get_csv(file_path)

main()