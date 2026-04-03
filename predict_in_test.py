import os
import re
import torch
import argparse
import csv
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.join(BASE_DIR, 'ChineseLyrics', 'test.txt.csv')
save_path = os.path.join(BASE_DIR, 'test_result', 'case.txt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_sentence(input, model, tokenizer):
    model.eval()                 
    input_ids = tokenizer.encode(input, return_tensors="pt").to(device)  
    outputs = model.generate(input_ids=input_ids,num_beams=20, max_length=128, repetition_penalty=10.0)
    output_str = tokenizer.decode(outputs.reshape(-1), skip_special_tokens=False)
    return output_str

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.path.join(BASE_DIR, 'checkpoints'))
    parser.add_argument("--model_name", type=str, default='mt5')
    args = parser.parse_args()
    return args

def load_test_data(path):
    with open(path,'r') as csvfile:
        reader = csv.DictReader(csvfile)
        src_list = [row['\ufeffsrc'] for row in reader]   # Keyword data from the src column
    return  src_list


if __name__ == '__main__':
    args = args()
    if args.model_name == 'mt5':
        model = MT5ForConditionalGeneration.from_pretrained(args.model_path).to(device)
        tokenizer = MT5Tokenizer.from_pretrained(args.model_path)
        
    task_prefix = 'trigger:'
    test_data = load_test_data(test_path)
    with open(save_path,"w") as w:
        for input in test_data:
            complete_input = task_prefix + input
            mode = r'[^a-z<>0123456789_▁]+'                 # Remove special tokens
            output = (generate_sentence(complete_input,model,tokenizer))
            mask_group = re.findall(mode,output)
            word_group = re.findall(mode,input)
            mask_group.remove(mask_group[0])
            mask_group.remove(mask_group[-1])
            complete_str = ''
            temp_i = 0
            try:
                for i in range(len(word_group)):
                    complete_str = complete_str +mask_group[i] + word_group[i]
                    temp_i = i + 1
                complete_str = complete_str + mask_group[temp_i]
                if temp_i < len(mask_group) - 1:
                    temp_i += 1
                    complete_str += mask_group[temp_i]
                w.write(complete_str.replace(" ",""))
                w.write('\n')
            except Exception as e:
                print(str(e))