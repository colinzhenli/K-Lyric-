import pandas as pd
import os
# Convert txt files to csv files

data_path = '../ChineseLyrics'


def get_csv(file):
    df = pd.read_csv(file,delimiter=";",names=['src','tgt'])  # ';' is the delimiter
    # df.columns = ['id','file','text']
    # encoding='utf_8_sig' prevents garbled characters when saving CSV
    df.to_csv(f"{file}.csv", encoding='utf_8_sig', index=False)
    return 


def main():
    for file in ['train.txt','valid.txt', 'test.txt']:
        file_path = os.path.join(data_path, file)
        get_csv(file_path)

main()