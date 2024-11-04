import os
from PIL import Image
import numpy as np
from tqdm.notebook import tqdm
import cv2 
from tacobox import Taco
import random
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from transformers import RobertaTokenizerFast, GPT2Tokenizer
from transformers import RobertaConfig, EncoderDecoderConfig, EncoderDecoderModel
from transformers import GPT2Config, GPT2LMHeadModel
from evaluate import load

# Load and prepare data
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            img_path, text = line.strip().split('\t')
            data.append({
                'file_name': os.path.join('', img_path),
                'text': text
            })
    return pd.DataFrame(data)

# Load datasets
train_df = load_data('train.txt')
eval_df = load_data('val.txt')
test_df = load_data('test.txt')

class HWRDataset(Dataset):
    def __init__(self, df, tokenizer, input_width=1600, 
                 input_height=64,
                 aug=False,
                 taco_aug_frac=0.9):
        self.df = df
        self.input_width = input_width
        self.input_height = input_height
        self.tokenizer = tokenizer
        self.mytaco = Taco(
            cp_vertical=0.2,
            cp_horizontal=0.25,
            max_tw_vertical=100,
            min_tw_vertical=10,
            max_tw_horizontal=50,
            min_tw_horizontal=10
        )
        self.aug = aug
        self.taco_aug_frac = taco_aug_frac

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx): 
        file_name = self.df['file_name'].iloc[idx]
        text = self.df['text'].iloc[idx]
    
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Unable to load image at {file_name}. Skipping.")
            return None
        pixel_values = self.preprocess(image, self.aug)

        try:
            labels = self.tokenizer(text).input_ids
        except Exception as e:
            print(f"Error tokenizing text '{text}': {e}")
            return None  # Skip this entry if an error occurs

    # Set padding and EOS token IDs properly
        labels = [self.tokenizer.bos_token_id] + [label if label != self.tokenizer.pad_token_id else -100 for label in labels] + [self.tokenizer.eos_token_id]
    
    # Create attention mask
        attention_mask = [1 if token != -100 else 0 for token in labels]  # -100 for ignored tokens

        encoding = (torch.tensor(pixel_values[None, :, :]).float(), torch.tensor(labels), torch.tensor(attention_mask))
    
        return encoding

    def preprocess(self, img, augment=True):
        if augment:
            img = self.apply_taco_augmentations(img)
        
        img = img/255
        img = img.swapaxes(-2,-1)[...,::-1]
        target = np.ones((self.input_width, self.input_height))
        new_x = self.input_width/img.shape[0]
        new_y = self.input_height/img.shape[1]
        min_xy = min(new_x, new_y)
        new_x = int(img.shape[0]*min_xy)
        new_y = int(img.shape[1]*min_xy)
        img2 = cv2.resize(img, (new_y,new_x))
        target[:new_x,:new_y] = img2
        return 1 - (target)

    def apply_taco_augmentations(self, input_img):
        random_value = random.random()
        if random_value <= self.taco_aug_frac:
            augmented_img = self.mytaco.apply_vertical_taco(
                input_img, 
                corruption_type='random'
            )
        else:
            augmented_img = input_img
        return augmented_img

def collate_fn(batch):
    src_batch, tgt_batch, attn_mask_batch = [], [], []
    batch_dict = {}
    
    for item in batch:
        if item is None:
            continue
        src_sample, tgt_sample, attn_mask = item
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        attn_mask_batch.append(attn_mask)

    src_batch = torch.stack(src_batch)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=-100)
    attn_mask_batch = pad_sequence(attn_mask_batch, batch_first=True, padding_value=0)

    batch_dict['pixel_values'] = src_batch
    batch_dict['labels'] = tgt_batch
    batch_dict['attention_mask'] = attn_mask_batch
    
    return batch_dict



# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("vocabs/ved/")
tokenizer.bos_token = '<s>'
tokenizer.eos_token = '</s>'
tokenizer.pad_token = '<pad>'
tokenizer.unk_token = '<unk>'

# Create datasets
train_dataset = HWRDataset(df=train_df, tokenizer=tokenizer, aug=True)
eval_dataset = HWRDataset(df=eval_df, tokenizer=tokenizer)
test_dataset = HWRDataset(df=test_df, tokenizer=tokenizer)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))
print("Number of test examples:", len(test_dataset))

# Model architecture
class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 4),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

    def forward(self, src: Tensor):
        src = self.conv1(src)
        src = self.conv2(src)
        src = self.conv3(src)
        src = self.conv4(src)
        src = self.conv5(src)
        src = src.squeeze(-1)
        src = src.permute((0, 2, 1)).contiguous()
        return src

def model_conv_transformer(vocab_size):
    model_conv = Conv()

    dec = {
        'vocab_size': vocab_size,
        'n_positions': 512,
        'n_embd': 256,
        'n_head': 4,
        'n_layer': 2
    }

    enc = {
        'vocab_size': vocab_size,
        'num_hidden_layers': 2,
        'hidden_size': 256,
        'num_attention_heads': 4,
        'intermediate_size': 1024,
        'hidden_act': 'gelu'
    }

    enc_config = RobertaConfig(**enc)
    dec_config = GPT2Config(**dec)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
    model_transformer = EncoderDecoderModel(config=config)

    return model_conv, model_transformer

# Setup device and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv, transformer = model_conv_transformer(vocab_size=tokenizer.vocab_size)

conv.to(device)
transformer.to(device)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=16, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

# Configure transformer
transformer.config.decoder_start_token_id = tokenizer.bos_token_id
transformer.config.pad_token_id = tokenizer.pad_token_id
transformer.config.vocab_size = transformer.config.decoder.vocab_size
transformer.config.eos_token_id = tokenizer.eos_token_id
transformer.config.max_length = 256
transformer.config.early_stopping = False
transformer.config.no_repeat_ngram_size = 0
transformer.config.length_penalty = 1
transformer.config.num_beams = 4
transformer.config.temperature = 1

# CER computation
cer = load("cer")
def compute_cer(pred_ids, label_ids):
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    combine = [(x, y) for x, y in zip(pred_str, label_str) if x]
    pred_str = [x for x, y in combine]
    label_str = [y for x, y in combine]

    cer_score = cer.compute(predictions=pred_str, references=label_str)
    return cer_score

# Training setup
params = list(conv.parameters()) + list(transformer.parameters())
optimizer = torch.optim.Adam(params, lr=0.0003, betas=(0.9, 0.98), eps=1e-9)

num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
num_eval_steps = num_epochs * len(eval_dataloader)

print("Total training steps:", num_training_steps)

# Training loop
progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(num_eval_steps))

for epoch in range(num_epochs):
    # Training
    conv.train()
    transformer.train()
    train_loss = 0.0
    eval_loss = 0.0
    correct_train = 0.0
    total_train = 0.0
    correct_eval = 0.0
    total_eval = 0.0
    
    for batch in train_dataloader:
        if batch is None:
            continue
            
        for k, v in batch.items():
            batch[k] = v.to(device)
            
        outputs = conv(batch['pixel_values'])
        labels = batch['labels']
        outputs = transformer(inputs_embeds=outputs, labels=labels)

        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        with torch.no_grad():
            preds = torch.argmax(logits, axis=-1)
            mask = torch.ones_like(labels).to(device)
            mask[labels==-100] = 0
            correct_train += ((preds == labels)*mask).sum()
            total_train += mask.sum()

        progress_bar_train.update(1)
        
    print(f"Train Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    print(f"Train ACC after epoch {epoch}:", correct_train/total_train)
    
    # Evaluation
    conv.eval()
    transformer.eval()
    valid_cer = 0.0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            if batch is None:
                continue
                
            for k, v in batch.items():
                batch[k] = v.to(device)

            outputs = conv(batch['pixel_values'])
            labels = batch['labels']
            outputs = transformer(inputs_embeds=outputs, labels=labels)
            
            logits = outputs.logits
            loss = outputs.loss
            eval_loss += loss.item()

            preds = torch.argmax(logits, axis=-1)
            mask = torch.ones_like(labels).to(device)
            mask[labels==-100] = 0
            correct_eval += ((preds == labels)*mask).sum()
            total_eval += mask.sum()

            progress_bar_eval.update(1)

        print(f"Val Loss after epoch {epoch}:", eval_loss/len(eval_dataloader))
        print(f"Val ACC after epoch {epoch}:", correct_eval/total_eval)

        # Save models
        os.makedirs("./conv_transformer_weights/icdar", exist_ok=True)
        transformer.save_pretrained("./conv_transformer_weights/icdar")
        torch.save(conv.state_dict(), "./conv_transformer_weights/icdar/conv.pt")

# Final evaluation
conv.eval()
transformer.eval()
test_cer = 0.0

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        if batch is None:
            continue
            
        outputs = conv(batch["pixel_values"].to(device))
        outputs = transformer.generate(inputs_embeds=outputs)
        error = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
        test_cer += error 

print("Test CER:", test_cer / len(test_dataloader))
