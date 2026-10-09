import torch
import pandas as pd
from sklearn import metrics
from transformers import BertTokenizer
from transformers import RobertaTokenizer

from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.models.deberta.modeling_deberta import DebertaForSequenceClassification

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data import TensorDataset, random_split
from sklearn.metrics import f1_score
import os
import argparse
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from safetensors.torch import load_model


seed=42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



pretrained_model_path='/pre-trained/' #pre-trained model path


finetuned_model_path='/finetuned/model.safetensors'

num_labels = 2


# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#lambd= 0.3
epsilon=0

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", default='RoBerta', help="Name of the model to be used")
parser.add_argument("--batchsize", "-b", help="batch size to be used", default=32)

args = parser.parse_args()
if args.model:
    modelName = args.model
batch_size = 0 
if args.batchsize:
    batch_size = int(args.batchsize)
print('batch_size...', batch_size, flush=True)


print('Model Name is:', modelName)

datasetName =''

if datasetName == 'RHMD':

    df_valid = pd.read_csv('../src/Dataset/rhmd/RHMD_3_Class_cleaned_test.csv', sep=",", encoding="utf-8",names=['texts','labels'],  usecols=range(2))
    num_labels= 3
    MAX_SEQ_LEN= 215


elif datasetName == 'dreaddit':

    df_valid = pd.read_csv('../src/Dataset/dreaddit/dreaddit-test-clean.csv', sep=",", encoding="utf-8", names = ['subreddit', 'texts','labels'],  usecols=range(3))
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

    if modelName == 'BERT':
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
            logits = outputs[1]

            _, preds = torch.max(outputs[1], dim=1)
            predictions.extend(preds)
            #prediction_probs.extend(probs)
            real_values.extend(targets)
            
            val_loss += loss.item()
            nb_val_steps += 1

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return  predictions,  real_values, val_loss / nb_val_steps







valid_sentences = df_valid.texts.str.lower().to_list()
valid_labels = df_valid.labels.to_list()


valid_input_ids, valid_attention_masks = Get_Encodings(modelName, valid_sentences)
valid_input_ids = torch.cat(valid_input_ids, dim=0)
valid_attention_masks = torch.cat(valid_attention_masks, dim=0)
valid_labels = torch.tensor(valid_labels)


test_dataset = TensorDataset(
            valid_input_ids, valid_attention_masks, valid_labels)




model = Get_Model(modelName)

load_model(model, finetuned_model_path)

model = model.to(device)

# For validation the order doesn't matter, so we'll just read them sequentially.
test_dataloader = DataLoader(
            test_dataset,  # The validation samples.
            sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
            batch_size=batch_size,num_workers=5  # Evaluate with this batch size.
        )
y_pred,  y_gt, _ = get_predictions(
        model,
        test_dataloader
    )

f1 =f1_score(y_pred, y_gt, average='macro')
print('f1_score macro:', str(f1),flush=True )

f1_mic =f1_score(y_pred, y_gt, average='micro')
print('f1_score micro:', str(f1_mic),flush=True )

precision =precision_score(y_pred, y_gt, average='macro')
precision_mic =precision_score(y_pred, y_gt, average='micro')

recall =recall_score(y_pred, y_gt, average='macro')
recall_mic =recall_score(y_pred, y_gt, average='micro')
print('precision macro score:', str(precision),flush=True )
print('precision micro score:', str(recall_mic),flush=True )

print('recall macro score:', str(recall),flush=True )
print('recall micro score:', str(recall_mic),flush=True )
