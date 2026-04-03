import jieba
import jieba.analyse
import random
corpus_path = '../ChineseLyrics/lyrics.txt'
save_path = '../ChineseLyrics/src_tgt_all.txt'
random.seed(0)

def organazation_src_tgt(corpus_path,save_path):
    with open(corpus_path,"r") as f, open(save_path,"w") as w:
        for line in f:
            line = line.strip()
            sen = line
            label = 'haha'
            random_num = random.randint(1,2)                # Randomly choose the number of keywords to extract
            keywords = jieba.analyse.textrank(sen,topK=random_num)           # TextRank works better than TF-IDF here
            if len(keywords) == 0:              # No keywords extracted
                continue
            true_order = []
            for keyword in keywords:
                index = sen.find(keyword)
                temp_tuple = (keyword,index)
                true_order.append(temp_tuple)
            true_order.sort(key = lambda elem:elem[1])      # Sort keywords by their original position in the sentence
            word_list = []
            for word,index in true_order:
                word_list.append(word)
            write_template(word_list,label,sen,w)

def write_template(word_list,label,sen,w):
    num = len(word_list)
    if num == 1:
        w.write("<a>{}<b>;".format(word_list[0]))   
    elif num == 2:
        w.write("<a>{}<b>{}<c>;".format(word_list[0],word_list[1]))
    
    write_tgt(word_list,sen,label,w)

def write_tgt(word_list,sen,label,w):
    begin = 0
    segments = []
    for word in word_list:
        index = sen.find(word)
        segment = sen[begin:index]
        begin = index + len(word)
        segments.append(segment)
    segments.append(sen[begin:])
    num = len(segments)
    if num == 2:
        print(f'<a>{segments[0]}<b>{segments[1]}', file=w)
    elif num == 3:
        print(f'<a>{segments[0]}<b>{segments[1]}<c>{segments[2]}', file=w)

          
    
    
    
    

if __name__ == '__main__':
    organazation_src_tgt(corpus_path,save_path)
    

            