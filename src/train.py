import torch
import pandas as pd
from sklearn.utils import shuffle
from sklearn import metrics
from collections import defaultdict
from transformers import BertTokenizer
from transformers import RobertaTokenizer

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification

import numpy as np
from torch.optim import Adam
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data import TensorDataset, random_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import os
import argparse
import gc 

seed=42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class EarlyStop:
    def __init__(self, mode="max", history=5):
        if mode == "max":
            self.best_func = np.max
            self.best_val = -np.inf 
            self.comp = lambda x, y: x >= y
        elif mode == "min":
            self.best_func = np.min
            self.best_val = np.inf 
            self.comp = lambda x, y: x <= y
        else:
            print("Please use 'max' or 'min' for mode.")
            quit()
        
        self.history_num = history
        self.history = np.zeros((self.history_num, ))
        self.total_num = 0

    def check(self, score):
        self.history[self.total_num % self.history_num] = score
        self.total_num += 1
        current_best_val = self.best_func(self.history)
        
        if self.total_num <= self.history_num:
            return False

        if self.comp(current_best_val, self.best_val):
            self.best_val = current_best_val
            return False
        else:
            return True



pretrained_model_path='/pre-trained/models/'
test_data_path=''
results_rootpath=''
modelName =''
datasetName=''




MAX_SEQ_LEN= 0
EPOCHS = 20
num_labels = 0


# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")




def barlow_loss_func(z1: torch.Tensor, z2: torch.Tensor, lamb: float = 5e-3, scale_loss: float = 0.025) -> torch.Tensor:
    
    N, D = z1.size()

    # to match the original code
    bn = torch.nn.BatchNorm1d(D, affine=False).to(z1.device)
    z1 = bn(z1)
    z2 = bn(z2)

    corr = torch.einsum("bi, bj -> ij", z1, z2) / N


    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    cdif[~diag.bool()] *= lamb
    loss = scale_loss * cdif.sum()
    return loss

class BarlowTwins(nn.Module):
    def __init__(
        self,args
    ):


        super().__init__()

        self.lamb = 0.0051
        self.scale_loss = 0.024

        # projector
        self.projector = nn.Sequential(
            nn.Linear(1024, 1024), #1024
            nn.BatchNorm1d(1024), #1024
            nn.ReLU(),
            nn.Linear(1024, 1024), #1024
            nn.BatchNorm1d(1024), #1024 
            nn.ReLU(),
            nn.Linear(1024, 300), #1024
        )



    def forward(self, z1, z2):

        z1 = self.projector(z1)
        z2 = self.projector(z2)        
        barlow_loss = barlow_loss_func(z1, z2, lamb=self.lamb, scale_loss=self.scale_loss)
        return barlow_loss

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", default='BERT', help="Name of the model to be used")
parser.add_argument("--ds", "-d", default='dreaddit', help="Name of the model to be used")
parser.add_argument("--batchsize", "-b", help="batch size to be used", default=16)
parser.add_argument("--alpha", "-a", help="batch size to be used", default=0.1)
parser.add_argument("--lr", "-l", help="learning rate to be used", default=1e-5)
parser.add_argument("--model1_layer", "-ml1", help="layer to be used", default=0)
parser.add_argument("--model2_layer", "-ml2", help="layer to be used", default=3)

args = parser.parse_args()
if args.model:
    modelName = args.model
if args.ds:
    datasetName = args.ds
batch_size = 0 
if args.batchsize:
    batch_size = int(args.batchsize)
    
alpha = 0.0
if args.alpha:
    alpha = float(args.alpha)
model1_layer = 0 
if args.model1_layer:
    model1_layer = int(args.model1_layer)
model2_layer = 0 
if args.model2_layer:
    model2_layer = int(args.model2_layer)
lr = 1e-5
if args.lr:
    lr = float(args.lr)
print('batch_size...', batch_size, flush=True)
print('Model Name is:', modelName)
print('alpha is:', str(alpha), flush=True)
print('lr is:', str(lr), flush=True)
print('model1_layer is:', str(model1_layer), flush=True)

print('model2_layer is:', str(model2_layer), flush=True)


if datasetName == 'RHMD':
    df_train = pd.read_csv('../src/Dataset/rhmd/RHMD_3_Class_cleaned_train.csv', sep=",", encoding="utf-8",names=['texts', 'labels'],  usecols=range(2))
    df_train = shuffle(df_train)

    df_valid = pd.read_csv('../src/Dataset/rhmd/RHMD_3_Class_cleaned_val.csv', sep=",", encoding="utf-8",names=['texts','labels'],  usecols=range(2))
    #print('df_valid before:', df_valid.head())
    num_labels= 3
    MAX_SEQ_LEN= 215


elif datasetName == 'dreaddit':
    df_train = pd.read_csv('../src/Dataset/dreaddit/dreaddit-train-clean.csv', sep=",", encoding="utf-8", names = ['subreddit','texts','labels'],  usecols=range(3))
    df_train = shuffle(df_train)

    df_valid = pd.read_csv('../src/Dataset/dreaddit/dreaddit-val-clean.csv', sep=",", encoding="utf-8", names = ['subreddit','texts','labels'],  usecols=range(3))
    #print('df_valid before:', df_valid.head())
    num_labels= 2
    MAX_SEQ_LEN= 256
    

    


save_model_name=''
reportname=''


def Get_Model(modelName):
    model=''
    if modelName == 'BERT':
        model = BertForSequenceClassification.from_pretrained(
            # Use the 12-layer BERT model, with an uncased vocab.
            pretrained_model_path+'bert-large/',
            # The number of output labels--2 for binary classification.
            num_labels=num_labels
        )
    elif modelName == 'RoBerta':
        model = RobertaForSequenceClassification.from_pretrained(
            # Use the 12-layer BERT model, with an uncased vocab.
            pretrained_model_path+'roberta-large',
            # The number of output labels--2 for binary classification.
            num_labels=num_labels
        )

    return model

def Get_Encodings(modelName, inputText):   
    input_ids = []
    attention_masks = []
    
    elif modelName == 'BERT':
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_path+'bert-large/', do_lower_case=True)
    elif modelName == 'RoBerta':
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_path+'roberta-large/', do_lower_case=True)

    for sent in inputText:   
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=MAX_SEQ_LEN,  # Pad & truncate all sentences.
            pad_to_max_length=True,

            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt')  # Return pytorch tensors.
        input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    return input_ids, attention_masks


def get_predictions(model, data_loader):
    model = model.eval()

    tweet_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    val_loss = 0
    nb_val_steps = 0

    with torch.no_grad():
        for d in data_loader:

            input_ids = d[0].to(device)
            attention_mask = d[1].to(device)
            targets = d[2].to(device)

            outputs = model(input_ids=input_ids,attention_mask=attention_mask, labels=targets)

            loss = outputs[0]

            _, preds = torch.max(outputs[1], dim=1)

            predictions.extend(preds)
            real_values.extend(targets)
            
            val_loss += loss.item()
            nb_val_steps += 1

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return  predictions,  real_values, val_loss / nb_val_steps




stopper = EarlyStop(mode="max", history=5)
def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples, alpha, model1_layer, model2_layer):
    model = model.train()
    losses = []
    acc = 0
    counter = 0
    tr_loss = 0
    nb_tr_steps = 0
    contr_tr_loss= 0



    for step, d in enumerate(data_loader):
        last_hidden_state_Normal =[]
        last_hidden_state_Pert= []

        input_ids = d[0].type(torch.LongTensor)
        attention_mask=  d[1].type(torch.FloatTensor)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = d[2].to(device)
        optimizer.zero_grad()
        
        outputs= model(input_ids=input_ids, attention_mask=attention_mask, labels=targets,output_hidden_states=True,)
        hidden_states1 = outputs.hidden_states[model1_layer]
        

        loss1= outputs.loss
        
        outputs2= model(input_ids=input_ids, attention_mask=attention_mask,  labels=targets, output_hidden_states=True, noise_layer =model2_layer, noise_amount= hidden_states1 )
        hidden_states2 = outputs2.hidden_states[model2_layer]

        loss2= outputs2.loss
        
        barlowTwins_Loss = barlowTwins.forward(hidden_states1.mean(dim=1), hidden_states2.mean(dim=1))
        barlowTwins_Loss =  barlowTwins_Loss *alpha
        
        loss= ((1-alpha)*(loss1 + loss2)/2) + barlowTwins_Loss*alpha
        
        
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


        tr_loss += loss.item()
        contr_tr_loss+=barlowTwins_Loss.item()
        if nb_tr_steps%40 == 0:
            print("Train batch loss1: {}".format(loss1.item()), "For step No:",nb_tr_steps, flush=True )
            print("Train batch loss2: {}".format(loss2.item()), "For step No:",nb_tr_steps, flush=True )
            print('Contrastive batch loss: ', barlowTwins_Loss.item(), "For step No:",nb_tr_steps ,flush=True)
        nb_tr_steps += 1
        


    return tr_loss / nb_tr_steps, contr_tr_loss / nb_tr_steps




train_sentences = df_train.texts.str.lower().to_list()

train_labels = df_train.labels.to_list()

valid_sentences = df_valid.texts.str.lower().to_list()
valid_labels = df_valid.labels.to_list()

input_ids, attention_masks = Get_Encodings(modelName, train_sentences)

train_input_ids = torch.cat(input_ids, dim=0)
train_attention_masks = torch.cat(attention_masks, dim=0)
train_labels = torch.tensor(train_labels)

valid_input_ids, valid_attention_masks = Get_Encodings(modelName, valid_sentences)
valid_input_ids = torch.cat(valid_input_ids, dim=0)
valid_attention_masks = torch.cat(valid_attention_masks, dim=0)
valid_labels = torch.tensor(valid_labels)



# Combine the training inputs into a TensorDataset.
train_dataset = TensorDataset(
            train_input_ids, train_attention_masks, train_labels)

valid_dataset = TensorDataset(
            valid_input_ids, valid_attention_masks, valid_labels) 



save_fine_tuned='/finetuned/' #save finetuned model
reportname=save_fine_tuned+"val_results.txt"


if not os.path.exists(save_fine_tuned):
    os.makedirs(save_fine_tuned)
    
model= Get_Model(modelName)

model = model.to(device)


barlowTwins = BarlowTwins(args).to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay':0.0}
]


train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=batch_size,num_workers=5  # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            valid_dataset,  # The validation samples.
            sampler=SequentialSampler(valid_dataset),  # Pull out batches sequentially.
            batch_size=batch_size,num_workers=5  # Evaluate with this batch size.
        )
optimizer = Adam(model.parameters(), lr=lr)


total_steps = len(train_dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
optimizer,
num_warmup_steps=0,
num_training_steps=total_steps)

best_loss = np.inf
best_val_f1 =0.0
for epoch in range(0,EPOCHS):
    print('Epoch No: ', str(epoch+1))


    train_loss, bt_loss = train_epoch(
                model,
                train_dataloader,
                optimizer,
                device,
                scheduler,
                len(df_train), alpha, model1_layer, model2_layer
            )

    print('Train loss:', str(train_loss),flush=True )
    print('Train Contrastive loss: ', str(bt_loss),flush=True)
    
    y_pred,  y_valid, val_loss = get_predictions(
            model,
            validation_dataloader
        )

    
    print('Validation loss:', str(val_loss),flush=True )
    f1 =f1_score(y_valid, y_pred, average='macro')
    print('Validation f1_score:', str(f1),flush=True )

    
    f1 =f1_score(y_valid, y_pred, average='macro')
    if f1 > best_val_f1:
                model.save_pretrained(save_fine_tuned)

                best_loss = val_loss
                
                best_val_f1 = f1
                print('best_val_f1: ', str(best_val_f1),flush=True)
                report = classification_report(y_valid, y_pred)
                with open(reportname, "w") as writer:
                    print("***** Test results *****")

                    #print(report,flush=True)
                    writer.write("\n\n")
                    writer.write(str(best_val_f1))
                    
                    writer.write("loss: ")
                    writer.write(str(val_loss))
    print('best_val_f1: ', str(best_val_f1),flush=True)
    if stopper.check(f1):
        print("Early Stopping at Epoch = ", str(epoch+1))
        break

    
    

