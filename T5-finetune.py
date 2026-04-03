import os
import torch
import wandb
import argparse
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EarlyStoppingCallback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'ChineseLyrics')

train_csv = os.path.join(DATA_DIR, 'train.txt.csv')
valid_csv = os.path.join(DATA_DIR, 'valid.txt.csv')
test_csv = os.path.join(DATA_DIR, 'test.txt.csv')

save_model_dir = os.path.join(BASE_DIR, 'checkpoints')

# Load model and tokenizer
tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
mt5model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')

max_input_length = 32
max_target_length = 128

task_prefix = 'trigger:'            # The actual prefix content does not matter

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_model_dir", type=str,default=save_model_dir)   # Path to save the model
    parser.add_argument("--batch_size_on_train", type=int, default=32)
    parser.add_argument("--batch_size_on_eval", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)     # Number of epochs to train
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)         # Stop training if loss does not decrease for this many epochs
    parser.add_argument("--wandb_running_name", type=str, default='test_mt5')
    parser.add_argument("--use_wandb", action='store_true', default=False)
    args = parser.parse_args()
    return args




def main(args):
    tokenizer.add_special_tokens({'additional_special_tokens':["<a>","<b>","<c>"]})  # Add custom tokens
    print(tokenizer.additional_special_tokens_ids) # Print the IDs of custom special tokens
    dataset = load_dataset('csv',data_files={'train':train_csv, 'validation':valid_csv, 'test':test_csv})   # Load dataset
    
    tokenizer_datasets = dataset.map(preprocess_function, batched=True)   # Apply preprocessing to all samples
    
    use_wandb = args.use_wandb
    if use_wandb:
        wandb.init(project="t5-cn")
    
    training_args = Seq2SeqTrainingArguments(                        
        output_dir=args.output_model_dir,
        overwrite_output_dir=True,
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size_on_train,
        per_device_eval_batch_size=args.batch_size_on_eval,
        weight_decay=0.01,
        save_total_limit=10,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        fp16=False,
        save_strategy='epoch', 
        dataloader_num_workers=args.num_workers,
        load_best_model_at_end=True,
        run_name=args.wandb_running_name if use_wandb else None,
        report_to='wandb' if use_wandb else 'none',
        logging_dir=os.path.join(BASE_DIR, 'logs'),
        generation_max_length=128,
        generation_num_beams=10,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    # patience = EarlyStoppingCallback(early_stopping_patience=args.patience)

    trainer = Seq2SeqTrainer(
        mt5model,
        training_args,
        train_dataset=tokenizer_datasets["train"],
        eval_dataset=tokenizer_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    trainer.train()         # Start fine-tuning

def preprocess_function(examples):          # Data preprocessing
    inputs = [task_prefix + line for line in examples['src']] 
    targets = [line for line in examples['tgt']]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding='longest')       # Encoder input

    with tokenizer.as_target_tokenizer():                   # Tokenizer for targets
        decoder_inputs = tokenizer(targets, max_length=max_target_length, truncation=True, padding='longest')       # This produces the decoder output; the model auto-shifts right and adds SOS token to get decoder input
        
    model_inputs['labels'] = decoder_inputs['input_ids']                # Do not rename the 'labels' key

    return model_inputs


if __name__ == '__main__':
    arg = args()
    main(arg)
    

