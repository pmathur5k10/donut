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

import torch
import numpy as np
import pandas as pd
import collections.abc as container_abcs
import transformers
from transformers import LayoutLMModel, LayoutLMForMaskedLM, LayoutLMForSequenceClassification, LayoutLMTokenizer, get_linear_schedule_with_warmup, LayoutLMv2Tokenizer, LayoutLMv2Processor, LayoutLMv2Model, LayoutLMv2FeatureExtractor
from sentence_transformers import SentenceTransformer, models
import torchtext
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from torch.nn import DataParallel
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import csv
import json
from PIL import Image
import string
from tqdm import tqdm
import torch.nn.functional as nnf
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
os.environ["TOKENIZERS_PARALLELISM"] = "false"

master_address = os.environ['MASTER_ADDR']
master_port = int(23457)
local_rank = int(os.environ.get('LOCAL_RANK') or 0)
world_size = int(os.environ['WORLD_SIZE'] or 1)
rank = int(os.environ['RANK'] or 0)
torch.distributed.init_process_group(backend='nccl')
print(f'native distributed, size: {world_size}, rank: {rank}, local rank: {local_rank}')
device = torch.device('cuda:' + str(local_rank))

##### Customize num_workers. Keep batch_size=1 ####

num_workers=2
batch_size=1
epochs=10
lr=1e-5
num_warm_steps=100
torch.manual_seed(0)
np.random.seed(0)

LayoutLMv2Tokenizer=LayoutLMv2Tokenizer.from_pretrained('microsoft/layoutlmv2-base-uncased')
additional_tokens=['<Widget>', '<TableRow>', '<NULL>', '<ChoiceGroup>', '<Footer>', '<Section>', '<ListItem>', '<Table>', '<TextRun>', '<TableCell>', '<TextBlock>', '<List>', '<Image>', '<Field>', '<form>', '<Header>', '<box>']
additional_token_words=['Widget', 'TableRow', 'NULL', 'ChoiceGroup', 'Footer', 'Section', 'ListItem', 'Table', 'TextRun', 'TableCell', 'TextBlock', 'List', 'Image', 'Field', 'form', 'Header', 'box']

category_map={}
for i in range(len(additional_token_words)):
    category_map[additional_token_words[i]]=i

LayoutLMv2Tokenizer.add_tokens(additional_tokens, special_tokens=True)
SentenceBERT = SentenceTransformer('paraphrase-MiniLM-L6-v2')

data_path="/home/code-base/scratch-space/Puneet/"
test_folder="Flamingo/training/"
processed_folder="processed_SentenceBERT/"
model_save_path="models/LayoutLMv2/"
input_image_path ="input/images/"

test=[]
test_fileList=os.listdir(data_path+test_folder+processed_folder)

for filename in test_fileList:
    test.append([filename, test_folder])
print(len(test))

empty=SentenceBERT.encode("")
sep=LayoutLMv2Tokenizer.encode(text=['[SEP]'], boxes=[[0,0,0,0]])[1]
pad=LayoutLMv2Tokenizer.encode(text=['[PAD]'], boxes=[[0,0,0,0]])[1]
class LayoutLMDataset(Dataset):
    def __init__(self, data):
        self.data=data
        self.num_samples=len(list(self.data))
        self.LayoutLMv2Tokenizer=LayoutLMv2Tokenizer
        self.feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)
        pass
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
             
        def normalize_boxes(box):
            
            if(box[0]>box[2]):
                temp=box[0]
                box[0]=box[2]
                box[2]=temp
            if(box[1]>=box[3]):
                temp=box[1]
                box[1]=box[3]
                box[3]=temp 
                
            for i in range(len(box)):
                if(box[i]<0):
                    box[i]=0
                elif(box[i]>1000):
                    box[i]=1000
            return box
        
        filename=self.data[idx][0]
        with open(data_path+self.data[idx][1]+processed_folder+filename) as f:
            data = json.load(f)
        
        height=data['height']
        width=data['width']
        
        candidate_parent_box=data['candidate_parent_box']
        list_child_boxes=data['list_child_boxes']
        candidate_parent_word="box"
        list_child_words=data['list_child_words']
        child_rel=data['rel']
        category=data['candidate_parent_word']
        image_file=data['fname']
        image = Image.open(data_path+self.data[idx][1]+input_image_path+image_file+".png").convert("RGB")
        encoding = self.feature_extractor(image, return_tensors="pt")
        pixel_values = encoding['pixel_values']

        list_normalized_child_boxes=list_child_boxes
        token_boxes = []
        input_ids=[]
        target=[]
        input_sentences=[]
        
        candidate_parent_box_mod=[max(0, candidate_parent_box[2]), max(0, candidate_parent_box[0]), max(0, candidate_parent_box[3]), max(0, candidate_parent_box[1])]

        candidate_parent_box_mod=normalize_boxes(candidate_parent_box_mod)

        word_tokens_candidate_parent = self.LayoutLMv2Tokenizer.encode(text=['<'+str(candidate_parent_word)+'>'], boxes=[candidate_parent_box_mod])
        parent_sentence=data['candidate_parent_sentence']

        input_ids.extend(word_tokens_candidate_parent)
        token_boxes.extend([candidate_parent_box_mod])
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[0, 0, 0, 0]]
        target.extend([-100, 0, -100])
        input_sentences.append(empty)
        input_sentences.append(parent_sentence)
        input_sentences.append(empty)
        for i in range(len(list_child_words)):
            word = list_child_words[i]
            normalized_child_boxes = list_normalized_child_boxes[i]
            normalized_child_boxes_mod=[max(0, normalized_child_boxes[2]), max(0, normalized_child_boxes[0]), max(0, normalized_child_boxes[3]), max(0, normalized_child_boxes[1])]
            
            normalized_child_boxes_mod=normalize_boxes(normalized_child_boxes_mod)
            child_sentence=data['list_child_sentences'][i]
            rel = child_rel[i]
            child_word_tokens = LayoutLMv2Tokenizer.encode(text=['<'+str(word)+'>'], boxes=[normalized_child_boxes_mod])[1]
            
            input_ids.append(child_word_tokens)
            input_sentences.append(child_sentence)
            token_boxes.extend([normalized_child_boxes_mod])
            target.append(rel)
        
        input_ids.append(sep)
        input_sentences.append(empty)
        token_boxes = token_boxes + [[1000, 1000, 1000, 1000]]
        target = target+[-100]
        
        l=len(input_ids)
        pad_token_id=pad
        input_ids.extend([pad_token_id]*(512-l))
        input_sentences.extend([empty]*(512-l))
        target.extend([-100]*(512-l))
        token_boxes.extend([[0, 0, 0, 0]]*(512-l))
        
        category_id=category_map[category]
        
        bbox = torch.tensor(token_boxes)
        target = torch.tensor(target).to(torch.float32)
        input_ids = torch.tensor(input_ids)
        category_id= torch.tensor(category_id)
        input_sentences=torch.tensor(input_sentences)
        input_sentences = torch.squeeze(input_sentences)
        
#         print(input_sentences.shape)
#         print(bbox.shape)
#         print(target.shape)
#         print(input_ids.shape)
#         print(pixel_values.shape)

        return {
            'input_ids':input_ids,
            'bbox': bbox,
            'image': pixel_values,
            'target': target,
            'category_id': category_id,
            'input_sentences': input_sentences
            }



def calculate_loss(embedding_dot_prod, parent_category, target, category, rank):
    weights = torch.tensor([0.003, 0.997]).to(rank)
    criterion_weighted = nn.CrossEntropyLoss(weight=weights)
    CEloss=nn.CrossEntropyLoss()
    target = target.long()

    loss_rel=criterion_weighted(embedding_dot_prod,target)*10 #[ 0, 1 , -100]
    loss_class=CEloss(parent_category,category)
    total_loss=loss_class+loss_rel
    return total_loss

class LayoutLMv2_Classification_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.LayoutLMv2Encoder = LayoutLMv2Model.from_pretrained('microsoft/layoutlmv2-base-uncased')
        self.LayoutLMv2Encoder.resize_token_embeddings(len(LayoutLMv2Tokenizer))
        self.num_classes=len(category_map)
        self.fc1=nn.Linear(769,self.num_classes)
        self.fc2=nn.Linear(1,2)
        self.fc_sentence=nn.Linear(384,1)

    def forward(self, input_ids, bbox, image, input_sentences):
        
        image = torch.squeeze(image,1)
        output=self.LayoutLMv2Encoder(input_ids=input_ids, bbox=bbox, image=image, output_hidden_states=True)
        output=output.hidden_states[-1][:,:512,:]
        input_sentences=self.fc_sentence(input_sentences)
        output=torch.cat((output, input_sentences),2)
        
        parent=output[:,1,:]
        parent=torch.unsqueeze(parent, 1)
        candidate_parent_embedding=torch.transpose(parent,1,2)
        embedding_dot_prod=torch.bmm(output,candidate_parent_embedding)
        embedding_dot_prod=self.fc2(embedding_dot_prod)
        embedding_dot_prod=torch.transpose(embedding_dot_prod,1,2)
        
        parent_category=self.fc1(parent)
        parent_category=torch.squeeze(parent_category,1)
        
        return parent_category,embedding_dot_prod

model=LayoutLMv2_Classification_model().to(rank)
model.load_state_dict(torch.load(data_path+model_save_path+ "LayoutLMv2_SentenceBERT_model_state.bin"))
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

target_dict={}
pred_dict={}

for i in range(len(additional_tokens)):
    target_dict[i]=[]
    pred_dict[i]=[]

def testing(testLoader, model, rank):
    sigmoid=torch.nn.Sigmoid()
    predicted_class=[]
    target_class=[]
    
    predicted_link=[]
    target_link=[]
    
    model = model.eval()
    with tqdm(testLoader, unit="batch") as tepoch:
        for data in tepoch:
            with torch.no_grad():
                input_ids = data['input_ids'].to(rank)
                bbox = data['bbox'].to(rank)
                image = data['image'].to(rank)
                target=data['target'].to(rank)
                category_id=data['category_id'].to(rank)
                input_sentences=data['input_sentences'].to(rank)
            
                parent_category, embedding_dot_prod = model(input_ids=input_ids,bbox=bbox, image=image, input_sentences=input_sentences)                
                softmax = nn.Softmax(dim=0)
                embedding_dot_prod=torch.squeeze(embedding_dot_prod, 0)
                link_prob=softmax(embedding_dot_prod)
                link_prob=torch.argmax(link_prob, dim=0)
                link_prob=link_prob.tolist()

                predicted_link.extend(link_prob)
                target=torch.squeeze(target, 0)
                target_link.extend(target.tolist())
                                
                prob = nnf.softmax(parent_category, dim = 1)
                top_p, top_class = prob.topk(1, dim = 1)
                predicted_class.append(top_class.item())
                target_class.append(category_id.item())

                index=category_id.item()

                target_dict[index].extend(target.tolist())
                pred_dict[index].extend(link_prob)
                
    return predicted_class, target_class, predicted_link, target_link

testSet=LayoutLMDataset(test)
testSampler = DistributedSampler(testSet, world_size, rank)
testLoader=DataLoader(testSet, sampler=testSampler, num_workers=num_workers, batch_size=batch_size)

predicted_class, target_class, predicted_link, target_link = testing(testLoader, model, rank)  
print("Class Metrics")
class_prediction_report = classification_report(target_class, predicted_class, output_dict=True)
with open('df_class_prediction_LMv2_SBERT.json', 'w') as fp:
    json.dump(class_prediction_report, fp)

target_link_mod=[]
predicted_link_mod=[]
for i in range(len(target_link)):
    if(target_link[i]!=-100):
        target_link_mod.append(target_link[i])
        predicted_link_mod.append(predicted_link[i])

print("Link Metrics")
rel_prediction_report = classification_report(target_link_mod, predicted_link_mod, output_dict=True)
with open('df_rel_prediction_LMv2_SBERT.json', 'w') as fp:
    json.dump(rel_prediction_report, fp)

comparative_list={}
for index in range(len(additional_tokens)):
    print("comparative", additional_tokens[index])
    target_mod=[]
    pred_mod=[]

    for i in range(len(target_dict[index])):
        if(target_dict[index][i]!=-100):
            target_mod.append(target_dict[index][i])
            pred_mod.append(pred_dict[index][i])
    try:
        df=classification_report(target_mod, pred_mod, output_dict=True)
        comparative_list[index]=df
    except:
        pass
with open('comparative_list_LMv2_SBERT.json', 'w') as fp:
    json.dump(comparative_list, fp)
