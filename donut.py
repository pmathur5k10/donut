import os
import pandas as pd
import seaborn as sns
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
import json
import csv
import torchtext
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
from torchtext.data import get_tokenizer
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from PIL import Image
import re
torch.cuda.empty_cache()

from transformers import AdamW, get_linear_schedule_with_warmup, Adafactor
from transformers import VisionEncoderDecoderConfig

max_length = 128
image_size = [256, 240]
# update image_size of the encoder
# during pre-training, a larger image size was used
# update max_length of the decoder (for generation)
config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
config.encoder.image_size = image_size # (height, width)
config.decoder.max_length = max_length

from transformers import DonutProcessor, VisionEncoderDecoderModel, BartConfig
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base", config=config)
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)

special_words_to_add={"additional_special_tokens": ["action_para", "component_para", "intial_state", "final_state"]}
# processor.tokenizer.add_special_tokens(special_words_to_add)
tokenizer=processor.tokenizer

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device_ids = [0,1]
# print(torch.cuda.memory_summary(device=None, abbreviated=False))

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

batch_size = 1
epochs = 1
learning_rate = 2e-5
warmup_steps = 20000
epsilon = 1e-8
sample_every = 100

dataset = "./DocEdit_Dataset/PDF/"

train_path = dataset + "train.csv"
val_path = dataset + "val.csv"
test_path = dataset + "test.csv"

train =[]
val =[]
test =[]

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = val_df

train_list = train_df.values.tolist()
val_list = val_df.values.tolist()
test_list = test_df.values.tolist()

for i in range(len(train_list)):
    train.append({'command':train_list[i][3], 'user_request': train_list[i][1], 'image':train_list[i][2]})

for i in range(len(val_list)):
    val.append({'command':val_list[i][3], 'user_request': val_list[i][1], 'image':val_list[i][2]})

for i in range(len(test_list)):
    test.append({'command':test_list[i][3], 'user_request': test_list[i][1], 'image':test_list[i][2]})


print(len(train), len(test), len(val))
train=train[:2]
val=val[:2]
test=test[:2]

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data
        self.tokenizer = tokenizer
        self.processor=processor
        self.max_length=max_length
        self.ignore_id = -100
        self.prompt_end_token=""
        self.prompt_end_token_id=self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        command_text = data['command']
        user_request_text = data['user_request']
        
        image_file = data['image']
        image_file = dataset+"Images/"+ image_file
        image = Image.open(image_file).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        task_prompt = "<s_docedit><s_question>{user_input}</s_question><s_answer>{command_output}</s_answer></s>"
        prompt = task_prompt.replace("{user_input}", user_request_text)
        prompt = prompt.replace("{command_output}", command_text)
        decoder_input_ids = self.tokenizer(prompt, add_special_tokens=False, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

        labels = decoder_input_ids.clone()
        labels=labels.squeeze(0) #.tolist()
        labels[labels == processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token
        labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id
        
        decoder_input_ids= decoder_input_ids.squeeze(0)
        # print(decoder_input_ids.shape, pixel_values.shape, labels.shape)

        return decoder_input_ids, pixel_values, labels

train_dataset = Dataset(train)
train_loader = DataLoader(train_dataset, batch_size, shuffle = False)

val_dataset = Dataset(val)
val_loader = DataLoader(val_dataset, batch_size, shuffle = False)

test_dataset = Dataset(test)
test_loader = DataLoader(test_dataset, batch_size, shuffle = False)

# model.decoder.resize_token_embeddings(len(tokenizer))
model = torch.nn.DataParallel(model, device_ids=device_ids)
model.to(device)

optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)

training_stats = []

for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_train_loss = 0
    model.train()

    for step, batch in enumerate(tqdm(train_loader)):

        decoder_input_ids = batch[0].to(device)
        pixel_values = batch[1].to(device)
        labels = batch[2].to(device)

        model.zero_grad()

        print(pixel_values.shape, decoder_input_ids.shape, labels.shape)

        outputs = model(pixel_values, decoder_input_ids=decoder_input_ids,labels=labels)      
        loss = outputs.loss
        loss = torch.mean(loss)
        
        batch_loss = loss.item()
        total_train_loss += batch_loss

        loss.backward()
        optimizer.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_loader)       
    
    print("Average training loss: {0:.2f}".format(avg_train_loss))
    print("Running Validation...")

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    for step, batch in enumerate(tqdm(val_loader)):
        
        decoder_input_ids = batch[0].to(device)
        pixel_values = batch[1].to(device)
        labels = batch[2].to(device)
        
        with torch.no_grad():
            outputs = model(pixel_values, decoder_input_ids=decoder_input_ids,labels=labels)                
            loss = outputs.loss
            loss = torch.mean(loss)
            
        batch_loss = loss.item()
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(val_loader)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
        }
    )

print("Training complete!")

output_dir = './donut_docedit_pdf_saved_models/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

model.eval()
ground_truth = []
prediction = []

for data in test:

    command_text = data['command']
    user_request_text = data['user_request']
    print("USR_REQ: ", user_request_text)
    image_file = data['image']
    image_file = dataset+"Images/"+ image_file
    image = Image.open(image_file).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()
    task_prompt = "<s_docedit><s_question>{user_input}</s_question><s_answer>"
    prompt = task_prompt.replace("{user_input}", user_request_text)
    decoder_input_ids = tokenizer(prompt, add_special_tokens=False, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

    decoder_input_ids = decoder_input_ids.to(device)
    pixel_values = pixel_values.to(device)
    
    outputs = model.module.module.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    answer = processor.token2json(sequence)['answer']
    print("Predicted: ", str(answer))
    prediction.append(str(answer.strip()))

    print("Ground Truth: ", str(command_text))
    ground_truth.append(str(command_text.strip()))

correct = 0
total = len(ground_truth)

for i in range(total):
    if ground_truth[i]==prediction[i]:
        correct = correct + 1

print((correct*100)/total)