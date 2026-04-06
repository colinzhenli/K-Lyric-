"""Compute PPL baselines: ground truth lyrics and random shuffle."""
import os
import csv
import random
import torch
import statistics
from transformers import BertTokenizer, GPT2LMHeadModel
from numpy import around

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

def cal_ppl_bygpt2(sens, tokenizer, model):
    inputs = tokenizer(sens, padding='max_length', max_length=50, truncation=True, return_tensors="pt").to(device)
    bs, sl = inputs['input_ids'].size()
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    logits = outputs[1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs['input_ids'][:, 1:].contiguous()
    shift_attentions = inputs['attention_mask'][:, 1:].contiguous()
    from torch.nn import CrossEntropyLoss
    loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
    meanloss = loss.sum(1) / shift_attentions.sum(1)
    ppl = torch.exp(meanloss).cpu().numpy().tolist()
    return ppl

def get_ground_truth_lyrics(test_csv_path):
    """Extract the original target lyrics from test CSV, stripping segment tokens."""
    targets = []
    with open(test_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row['\ufeffsrc']
            tgt = row['tgt']
            keywords = []
            for token in ['<a>', '<b>', '<c>']:
                src = src.replace(token, '\x00')
                tgt = tgt.replace(token, '\x00')
            src_parts = [p for p in src.split('\x00') if p]
            tgt_parts = [p for p in tgt.split('\x00') if p]
            reconstructed = ''
            ki = 0
            for i, seg in enumerate(tgt_parts):
                reconstructed += seg
                if ki < len(src_parts):
                    reconstructed += src_parts[ki]
                    ki += 1
            targets.append(reconstructed.strip())
    return targets

def make_random_baseline(lyrics):
    """Shuffle characters within each lyric to create a random baseline."""
    random.seed(42)
    shuffled = []
    for lyric in lyrics:
        chars = list(lyric)
        random.shuffle(chars)
        shuffled.append(''.join(chars))
    return shuffled

def compute_all_ppls(sentences, tokenizer, model, label=""):
    all_ppls = []
    batches = list_split(sentences, 100)
    for i, batch in enumerate(batches):
        batch = [s for s in batch if s.strip()]
        if not batch:
            continue
        ppls = cal_ppl_bygpt2(batch, tokenizer, model)
        all_ppls.extend(ppls)
        if (i + 1) % 10 == 0:
            print(f"  {label}: {i+1}/{len(batches)} batches done")
    return all_ppls

def print_stats(name, ppls):
    ppls_clean = [p for p in ppls if p < 10000]
    print(f"\n=== {name} ===")
    print(f"  Count:  {len(ppls_clean)}")
    print(f"  Mean:   {statistics.mean(ppls_clean):.2f}")
    print(f"  Median: {statistics.median(ppls_clean):.2f}")
    print(f"  Stdev:  {statistics.stdev(ppls_clean):.2f}")
    pct_under_80 = sum(1 for p in ppls_clean if p < 80) / len(ppls_clean) * 100
    print(f"  % < 80: {pct_under_80:.1f}%")
    return statistics.mean(ppls_clean), statistics.median(ppls_clean), pct_under_80

if __name__ == '__main__':
    test_csv = os.path.join(BASE_DIR, 'ChineseLyrics', 'test.txt.csv')

    print("Loading GPT-2 model...")
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").to(device)
    model.eval()

    print("Loading generated results...")
    gen_ppls = []
    with open(os.path.join(BASE_DIR, 'test_result', 'case_ppl.txt')) as f:
        for line in f:
            gen_ppls.append(float(line.strip()))

    print("Extracting ground truth lyrics from test set...")
    gt_lyrics = get_ground_truth_lyrics(test_csv)
    print(f"  Got {len(gt_lyrics)} ground truth lyrics")
    print(f"  Examples: {gt_lyrics[:3]}")

    print("\nComputing ground truth PPL...")
    gt_ppls = compute_all_ppls(gt_lyrics, tokenizer, model, label="GT")

    print("Creating random shuffle baseline...")
    random_lyrics = make_random_baseline(gt_lyrics)
    print(f"  Examples: {random_lyrics[:3]}")

    print("\nComputing random baseline PPL...")
    rand_ppls = compute_all_ppls(random_lyrics, tokenizer, model, label="Random")

    print("\n" + "="*60)
    gen_stats = print_stats("K-Lyric (Our Model)", gen_ppls)
    gt_stats = print_stats("Ground Truth Lyrics", gt_ppls)
    rand_stats = print_stats("Random Shuffle Baseline", rand_ppls)

    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'Method':<30} {'Mean PPL':>10} {'Median PPL':>12} {'% < 80':>8}")
    print("-" * 62)
    print(f"{'Random Shuffle':<30} {rand_stats[0]:>10.1f} {rand_stats[1]:>12.1f} {rand_stats[2]:>7.1f}%")
    print(f"{'K-Lyric (Ours)':<30} {gen_stats[0]:>10.1f} {gen_stats[1]:>12.1f} {gen_stats[2]:>7.1f}%")
    print(f"{'Ground Truth Lyrics':<30} {gt_stats[0]:>10.1f} {gt_stats[1]:>12.1f} {gt_stats[2]:>7.1f}%")

    with open(os.path.join(BASE_DIR, 'test_result', 'gt_ppl.txt'), 'w') as f:
        for p in gt_ppls:
            f.write(f"{around(p, 3)}\n")
    with open(os.path.join(BASE_DIR, 'test_result', 'random_ppl.txt'), 'w') as f:
        for p in rand_ppls:
            f.write(f"{around(p, 3)}\n")

    print("\nSaved to test_result/gt_ppl.txt and test_result/random_ppl.txt")
