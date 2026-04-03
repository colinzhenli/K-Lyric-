import json
import os

data_dir = '../ChineseLyrics'
save_path = '../ChineseLyrics/lyrics.txt'

def json2txt(file_path):
    sen_list = []
    with open(file_path, "r", encoding='utf-8') as f:
        json_data = json.load(f)
        for song in json_data:
            lyric = song['lyric']
            for sen in lyric:
                flag = is_chinese_in(sen)                      
                if flag and len(sen) >= 8 and len(sen) <= 20:  # Filter by length to remove noise
                    sen_list.append(sen)   
    return sen_list

def is_chinese_in(sen):       # Filter Chinese lyrics from the song data
    for ch in sen:
        if '\u4e00' <= ch <= '\u9fff':
            return True
        
    return False

def main():
    all_sen = []
    for path in ['lyrics1.json','lyrics2.json','lyrics3.json','lyrics4.json','lyrics5.json']:
        file_path = os.path.join(data_dir,path)
        sen_list = json2txt(file_path)
        all_sen.append(sen_list)
        with open(save_path,"w", encoding='utf-8') as w:
            for sen_list in all_sen:
                for sen in sen_list:
                    w.write(sen)
                    w.write('\n')


main()