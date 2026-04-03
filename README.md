# K-Lyric: Chinese Lyric Generation with mT5

K-Lyric is a Chinese lyric generation system that fine-tunes [google/mt5-small](https://huggingface.co/google/mt5-small) in a sequence-to-sequence setup. Given 1-3 keywords as input, the model generates corresponding Chinese lyric fragments.

## Environment Setup

### Prerequisites

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- (Optional) NVIDIA GPU with CUDA for faster training and inference

### Create Conda Environment

```bash
conda create -n klyric python=3.9 -y
conda activate klyric
```

### Install PyTorch

Install PyTorch with CUDA support (adjust the CUDA version to match your system):

```bash
# CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# CPU only (if no GPU available)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
K-Lyric/
├── T5-finetune.py            # Fine-tune mT5 for lyric generation
├── predicter.py              # Single-example inference demo
├── predict_in_test.py        # Batch inference on test set
├── cal_ppl_with_gpt2.py      # Perplexity evaluation using Chinese GPT-2
├── utils.py                  # Custom tokenizer utilities
├── requirements.txt          # Python dependencies
├── ChineseLyrics/            # Training data
│   ├── lyrics.txt            # Raw lyrics corpus
│   ├── src_tgt_all.txt       # Source-target pairs
│   ├── train.txt / .csv      # Training split
│   ├── valid.txt / .csv      # Validation split
│   └── test.txt / .csv       # Test split
├── prepare_data/             # Data pipeline scripts
│   ├── json2txt.py           # Convert JSON lyrics to text
│   ├── build_src_target.py   # Extract keywords and build src-tgt pairs
│   ├── split_data.py         # Split data into train/valid/test
│   └── txt2csv.py            # Convert text files to CSV
├── test_result/              # Inference output directory
│   ├── case.txt              # Generated lyrics
│   └── case_ppl.txt          # Perplexity scores
└── checkpoints/              # Saved model checkpoints (created during training)
```

## Data Preparation

The repo already includes pre-processed data in `ChineseLyrics/`. If you need to rebuild from raw JSON sources, run the pipeline in order:

```bash
cd prepare_data

# Step 1: Convert JSON lyrics files to text (requires lyrics1.json - lyrics5.json)
python json2txt.py

# Step 2: Extract keywords and build source-target pairs
python build_src_target.py

# Step 3: Split into train/valid/test sets
python split_data.py

# Step 4: Convert text files to CSV for Hugging Face datasets
python txt2csv.py
```

## Training

Fine-tune the mT5 model on the lyric generation task:

```bash
python T5-finetune.py
```

### Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--output_model_dir` | `./checkpoints` | Directory to save model checkpoints |
| `--batch_size_on_train` | `32` | Training batch size per device |
| `--batch_size_on_eval` | `4` | Evaluation batch size per device |
| `--num_workers` | `4` | Number of dataloader workers |
| `--num_epochs` | `100` | Number of training epochs |
| `--lr` | `1e-5` | Learning rate |
| `--patience` | `5` | Early stopping patience (currently disabled) |
| `--wandb_running_name` | `test_mt5` | Wandb run name |
| `--use_wandb` | `False` | Enable Weights & Biases logging |

### Example

```bash
# Train without wandb logging
python T5-finetune.py --num_epochs 50 --batch_size_on_train 16

# Train with wandb logging
python T5-finetune.py --use_wandb --wandb_running_name my_experiment
```

## Inference

### Single Example

Run inference on a single hardcoded example:

```bash
python predicter.py --model_path ./checkpoints
```

### Batch Inference on Test Set

Generate lyrics for all examples in the test set:

```bash
python predict_in_test.py --model_path ./checkpoints
```

Results are saved to `test_result/case.txt`.

## Evaluation

Compute perplexity of generated lyrics using a Chinese GPT-2 model:

```bash
python cal_ppl_with_gpt2.py
```

This reads `test_result/case.txt` and writes perplexity scores to `test_result/case_ppl.txt`.

## Input/Output Format

**Input**: Keywords wrapped in special markers, prefixed with `trigger:`

```
trigger:<a>keyword1<b>keyword2<c>
```

**Output**: Lyric fragments with the same marker structure, where text segments fill around the keywords.
