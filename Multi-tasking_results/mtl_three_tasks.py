# Importing Libraries

import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from transformers import BertModel, BertConfig
from transformers import XLMRobertaModel
from transformers import XLMRobertaTokenizer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from copy import deepcopy
import logging
import time
import os

# saving to log files

logging.basicConfig(filename=f'./logs/mBERT_{time.asctime().replace(" ","_")}.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a stream handler to print log messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

#Arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='no. of epochs' )
parser.add_argument('--bsz', type=int, default=32, help='batch size' )
parser.add_argument('--seqlen', type=int, default=128, help='sequence length' )
args = parser.parse_args()


# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Loading dataset

dt=pd.read_csv('/data1/aakash/Codemix/Aakash_02/Datasets/1HateSpeech_Codemix.csv') #codemix
dt=dt.dropna()

# train_df, remaining_df = train_test_split(dt, test_size=0.3, random_state=random_seed, stratify=dt['Tag'])
# test_df, val_df = train_test_split(remaining_df, test_size=0.5, random_state=random_seed, stratify=remaining_df['Tag'])
dy = pd.read_csv('/data1/aakash/Codemix/Aakash_02/Datasets/1Hatespeech_English(new).csv')



dx = pd.read_csv('/data1/aakash/Codemix/Aakash_02/Datasets/1HateSpeech_Hindi.csv')
hate_com = pd.concat([dt,dy,dx])
hate_com = hate_com.reset_index(drop=True)
hate_com
z = hate_com['Tag'].replace({1:999, 0:999})
z
# Loading dataset

dn=pd.read_csv('/data1/aakash/Codemix/Aakash_02/Datasets/1Humour_Codemix.csv') #codemix
dn=dn.dropna()


dm = pd.read_csv('/data1/aakash/Codemix/Aakash_02/Datasets/1Humour_English(NEW).csv')
dm = dm[:3000]
sar_com = pd.concat([dn,dm])
sar_com = sar_com.reset_index(drop=True)
sar_com
k=sar_com['Tag']
k
sar_com['Tag']= sar_com['Tag'].replace({1:999, 0:999})
sar_com
n = sar_com['Tag'].replace({1:999, 0:999})
n
df = pd.read_csv('/data1/aakash/Codemix/Aakash_02/Datasets/1Sarcasm_Codemix.csv')

train_df, remaining_df = train_test_split(df, test_size=0.3, random_state=random_seed, stratify=df['Tag'])
test_df, val_df = train_test_split(remaining_df, test_size=0.5, random_state=random_seed, stratify=remaining_df['Tag'])


dz = pd.read_csv('/data1/aakash/Codemix/Aakash_02/Datasets/1Sarcasm_English.csv')
dz = dz[:3000]

# Train Data

hum_com = pd.concat([train_df,dz])
hum_com = hum_com.reset_index(drop=True)
hum_com
y=hum_com['Tag']
y
s=pd.concat([z,n,y])
s=s.reset_index(drop=True)
s
hum_com['Tag']= hum_com['Tag'].replace({1:999, 0:999})
hum_com
r = hum_com['Tag'].replace({1:999, 0:999})
r
x=pd.concat([z,k,r])
x=x.reset_index(drop=True)
x
com_1 = pd.concat([hate_com,sar_com,hum_com])
com_1 = com_1.reset_index(drop=True)
com_1
com_1['Task2'] = s
com_1
com_1['Task3'] = x
com_1
# Rename the column
com_1 = com_1.rename(columns={'Tag': 'Task1'})
com_1
# Shuffle the DataFrame
data_tr = com_1.sample(frac=1, random_state=42)
data_tr.reset_index(drop=True, inplace=True)
data_tr

# Validation Data

data_val = val_df
data_val = data_val.reset_index(drop=True)
data_val
# Rename the column
data_val = data_val.rename(columns={'Tag': 'Task1'})
data_val
data_val['Task2'] = 999
data_val
# Interchange columns
data_val['Task1'], data_val['Task2'] = data_val['Task2'].copy(), data_val['Task1'].copy()
data_val
data_val['Task3'] = 999
data_val

# Test Data

data_test = test_df
data_test = data_test.reset_index(drop=True)
data_test
# Rename the column
data_test = data_test.rename(columns={'Tag': 'Task1'})
data_test
data_test['Task2'] = 999
data_test
# Interchange columns
data_test['Task1'], data_test['Task2'] = data_test['Task2'].copy(), data_test['Task1'].copy()
data_test
data_test['Task3'] = 999
data_test
class SharedCrossTaskModel(nn.Module):
    def __init__(self, num_classes_task1, num_classes_task2, num_classes_task3):
        super(SharedCrossTaskModel, self).__init__()

        # Load a pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased',output_hidden_states=True)
        #self.bert = XLMRobertaModel.from_pretrained('xlm-roberta-base',output_hidden_states=True)

        # Freeze all layers except the top 2
        for param in self.xlm_roberta.parameters():
            param.requires_grad = False

        # Unfreeze the parameters of the top 2 layers
        for param in self.xlm_roberta.encoder.layer[-2:].parameters():
            param.requires_grad = True

        # Task-specific layers for Task 1
        self.task1_specific_layers = nn.ModuleList([deepcopy(self.bert.encoder.layer[i]) for i in range(-4, 0, 1)])


        # Task-specific layers for Task 2
        self.task2_specific_layers = nn.ModuleList([deepcopy(self.bert.encoder.layer[i]) for i in range(-4, 0, 1)])


        # Task-specific layers for Task 3
        self.task3_specific_layers = nn.ModuleList([deepcopy(self.bert.encoder.layer[i]) for i in range(-4, 0, 1)])


        # setting the final gating mechanism
        self.gating_modules_task1 = Gating(2, self.bert.config.hidden_size)
        self.gating_modules_task2 = Gating(2, self.bert.config.hidden_size)
        self.gating_modules_task3 = Gating(2, self.bert.config.hidden_size)


        self.num_classes_task1 = num_classes_task1
        self.num_classes_task2 = num_classes_task2
        self.num_classes_task3 = num_classes_task3



        # Task-specific classifiers
        self.classifier_task1 = nn.Linear(self.bert.config.hidden_size, num_classes_task1)
        self.classifier_task2 = nn.Linear(self.bert.config.hidden_size, num_classes_task2)
        self.classifier_task3 = nn.Linear(self.bert.config.hidden_size, num_classes_task3)

    def forward(self, input_ids, attention_mask):
        
        """ 
        shape of input_ids: [batch_size, seq_len]
        shape of attention_mask: [batch_size, seq_len]
        """

        

        # BERT Output
        bert_output = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        hidden_states = bert_output.hidden_states

        #print(hidden_states[:8])


        # Task-specific layers for Task 1
        current_output1 = hidden_states[7]
       
        for layer in self.task1_specific_layers:
            current_output1 = layer(current_output1)[0]


        # Task-specific layers for Task 2
        current_output2 = hidden_states[7]
        

        for layer in self.task2_specific_layers:
            current_output2 = layer(current_output2)[0]



       # Task-specific layers for Task 3
        current_output3 = hidden_states[7]
        

        for layer in self.task3_specific_layers:
            current_output3 = layer(current_output3)[0]     


        # print(current_output1.shape)
        # print(bert_output.last_hidden_state[-4:].shape)
        # Combined output after gating

        combined_output_task1 = self.gating_modules_task1(tuple([hidden_states[-1], current_output1]))
        combined_output_task2 = self.gating_modules_task2(tuple([hidden_states[-1], current_output2]))
        combined_output_task3 = self.gating_modules_task3(tuple([hidden_states[-1], current_output3]))


        pooled_output_task1= combined_output_task1[:,0,:]
        pooled_output_task1=torch.squeeze(pooled_output_task1,dim=1)

        pooled_output_task2= combined_output_task2[:,0,:]
        pooled_output_task2=torch.squeeze(pooled_output_task2,dim=1)


        pooled_output_task3= combined_output_task3[:,0,:]
        pooled_output_task3=torch.squeeze(pooled_output_task3,dim=1)
        

        # Task-specific classifiers
        logits_task1 = self.classifier_task1(pooled_output_task1)
        logits_task2 = self.classifier_task2(pooled_output_task2)
        logits_task3 = self.classifier_task2(pooled_output_task3)
        
        return logits_task1, logits_task2, logits_task3
    
        
class Gating(torch.nn.Module):
    def __init__(self, num_gates, input_dim):
        super(Gating, self).__init__()
        self.num_gates = num_gates
        self.input_dim = input_dim
        if self.num_gates == 2:
            self.linear = torch.nn.Linear(self.num_gates * self.input_dim, self.input_dim)
        elif self.num_gates > 2:
            self.linear = torch.nn.Linear(self.num_gates * self.input_dim, self.num_gates * self.input_dim)
            self.softmax = torch.nn.Softmax(-1)
        else:
            raise ValueError('num_gates should be greater or equal to 2')

    def forward(self, tuple_of_inputs):
        # output size should be equal to the input sizes
        if self.num_gates == 2:
            #print("Tensor Sizes Before Concatenation:", tuple_of_inputs[0].size(), tuple_of_inputs[1].size())
            alpha = torch.sigmoid(self.linear(torch.cat(tuple_of_inputs, dim=-1)))
            output = torch.mul(alpha, tuple_of_inputs[0]) + torch.mul(1 - alpha, tuple_of_inputs[1])
        else:  # elif self.num_gates > 2:
            # extend the gating mechanism to more than 2 encoders
            batch_size, len_size, dim_size = tuple_of_inputs[0].size()
            alpha = torch.sigmoid(self.linear(torch.cat(tuple_of_inputs, dim=-1)))
            alpha = self.softmax(alpha.view(batch_size, len_size, dim_size, self.num_gates))
            output = torch.sum(torch.mul(alpha, torch.stack(tuple_of_inputs, dim=-1)), dim=-1)
        return output
    


# Instantiate the model
model = SharedCrossTaskModel(num_classes_task1 = 2, num_classes_task2 = 2, num_classes_task3 = 2)

model.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
#tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')


class CustomDataset(Dataset):
    def __init__(self, sentences, labels1, labels2, labels3, tokenizer, max_len):
        self.sentences = sentences
        self.labels1 = labels1
        self.labels2 = labels2
        self.labels3 = labels3
        self.tokenizer = tokenizer
        self.max_len = max_len
        #self.task_tokens = task_tokens

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        try:
            sentence = str(self.sentences[idx])
            label1 = self.labels1[idx]
            label2 = self.labels2[idx]
            label3 = self.labels3[idx]

            # Tokenize the input sentence with task tokens added as special tokens
            encoding = self.tokenizer.encode_plus(
                sentence,  
                add_special_tokens=True,
                max_length=self.max_len,  # Adjust max length without subtracting task tokens
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label1': torch.tensor(label1, dtype=torch.long),
                'label2': torch.tensor(label2, dtype=torch.long),
                'label3': torch.tensor(label3, dtype=torch.long)
            }
        except Exception as e:
            print(f"Problematic sentence: {self.sentences[idx]}")
            print(f"Problematic label1: {self.labels1[idx]}")
            print(f"Problematic label2: {self.labels2[idx]}")
            print(f"Problematic label3: {self.labels3[idx]}")
            raise e


# max_len = 128 
# batch_size = 64
# epochs = 50
# learning_rate = 5e-3
max_len = args.seqlen 
batch_size = args.bsz
epochs = args.epochs
learning_rate = args.lr

train_dataset = CustomDataset(data_tr['Sentence'].values, data_tr['Task1'].values, data_tr['Task2'].values, data_tr['Task3'].values, tokenizer, max_len)
val_dataset = CustomDataset(data_val['Sentence'].values, data_val['Task1'].values, data_val['Task2'].values, data_val['Task3'].values, tokenizer, max_len)
test_dataset = CustomDataset(data_test['Sentence'].values, data_test['Task1'].values, data_test['Task2'].values, data_test['Task3'].values, tokenizer, max_len)

# Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

# Loss

def custom_joint_loss(logits_task1, logits_task2, logits_task3, labels_task1, labels_task2, labels_task3):
    # Calculate the loss for task 2, but only for rows where task1 == 999 and task3 == 999
    mask_task2 = (labels_task1 == 999) & (labels_task3 == 999)
    loss_task2 = nn.CrossEntropyLoss()(logits_task2[mask_task2], labels_task2[mask_task2])

    # Calculate the loss for task 1, but only for rows where task2 == 999 and task3 == 999
    mask_task11 = (labels_task2 == 999)
    mask_task1 = (labels_task2 == 999) & (labels_task3 == 999)
    if torch.sum(mask_task11) == 0:
        loss_task1_2 = 0.0  # Set the loss to 0 when there are no rows with task2 == 999 
    else:
        loss_task1_2 = nn.CrossEntropyLoss()(logits_task1[mask_task1], labels_task1[mask_task1])

    # Calculate the loss for task 3, but only for rows where task1 == 999 and task2 == 999
    mask_task3 = (labels_task1 == 999) & (labels_task2 == 999)
    if torch.sum(mask_task11) == 0:
        loss_task3 = 0.0
    else:
        loss_task3 = nn.CrossEntropyLoss()(logits_task3[mask_task3], labels_task3[mask_task3])

    # Combine the losses
    joint_loss = loss_task2 + loss_task1_2 + loss_task3

    return joint_loss

#criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

train_losses = []
val_losses = []
best_val_loss = float('inf')
best_val_f1_score = 0
patience = 4  # Number of epochs to wait for improvement
counter = 0  # Counter to keep track of epochs without improvement


#best_model_save_path
tempdir = '/data1/aakash/Codemix/Aakash_02/.model/'
best_model_params_path = os.path.join(tempdir, f"mbert_Com.pt")


for epoch in range(epochs):
    
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_task1 = batch['label1'].to(device)
        labels_task2 = batch['label2'].to(device)
        labels_task3 = batch['label3'].to(device)
        

        optimizer.zero_grad()

        logits_task1, logits_task2, logits_task3 = model(input_ids, attention_mask=attention_mask)  # Get logits for all tasks

        loss = custom_joint_loss(logits_task1, logits_task2, logits_task3, labels_task1, labels_task2, labels_task3)  # Use custom joint loss

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)

    print(f'Epoch {epoch + 1}/{epochs}')
    logging.info(f'Train loss: {avg_train_loss:.4f}')

    train_losses.append(avg_train_loss)


    # Validation

    #criterion = nn.CrossEntropyLoss()

    model.eval()
    val_predictions_task1 = []
    val_predictions_task2 = []
    val_predictions_task3 = []
    val_labels_task1 = []
    val_labels_task2 = []
    val_labels_task3 = []
    val_loss = 0

    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_task1 = batch['label1'].to(device)
        labels_task2 = batch['label2'].to(device)
        labels_task3 = batch['label3'].to(device)

        with torch.no_grad():
            logits_task1, logits_task2, logits_task3 = model(input_ids=input_ids, attention_mask=attention_mask)

        predicted_labels_task1 = torch.argmax(logits_task1, dim=1)
        predicted_labels_task2 = torch.argmax(logits_task2, dim=1)
        predicted_labels_task3 = torch.argmax(logits_task3, dim=1)

        val_predictions_task1.extend(predicted_labels_task1.detach().cpu().numpy())
        val_predictions_task2.extend(predicted_labels_task2.detach().cpu().numpy())
        val_predictions_task3.extend(predicted_labels_task3.detach().cpu().numpy())
        val_labels_task1.extend(labels_task1.detach().cpu().numpy())
        val_labels_task2.extend(labels_task2.detach().cpu().numpy())
        val_labels_task3.extend(labels_task3.detach().cpu().numpy())

        # Use custom joint loss for validation loss calculation
        val_loss += custom_joint_loss(logits_task1, logits_task2, logits_task3, labels_task1, labels_task2, labels_task3).item()


    epoch_val_accuracy = accuracy_score(val_labels_task2, val_predictions_task2)
    avg_val_loss = val_loss / len(val_dataloader)
    #print(f'Validation accuracy: {epoch_val_accuracy:.4f}')
    logging.info(f'Validation loss: {avg_val_loss:.4f}')

    val_f1_score = f1_score(val_labels_task2, val_predictions_task2, average='macro')

    classification_report_epoch = classification_report(val_labels_task2, val_predictions_task2)
    logging.info(f'Classification Report per Epoch {epoch+1}:')
    logging.info(classification_report_epoch)
    
    val_losses.append(avg_val_loss)

    # Early stopping

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_params_path) # Saving best model
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    # if val_f1_score > best_val_f1_score:
    #     best_val_f1_score = val_f1_score
    #     torch.save(model.state_dict(), best_model_params_path)
    #     counter = 0
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print(f'Early stopping at epoch {epoch + 1}')
    #         break

    scheduler.step()

total_eval_accuracy = 0

test_predictions_task1 = []
test_predictions_task2 = []
test_predictions_task3 = []
true_labels_task1 = []
true_labels_task2 = []
true_labels_task3 = []

model.eval()

for batch in test_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels_task1 = batch['label1'].to(device)
    labels_task2 = batch['label2'].to(device)
    labels_task3 = batch['label3'].to(device)


    with torch.no_grad():
        logits_task1, logits_task2, logits_task3 = model(input_ids=input_ids, attention_mask=attention_mask)

    #logits = outputs.logits
    predicted_lab_task1 = torch.argmax(logits_task1, dim=1)
    predicted_lab_task2 = torch.argmax(logits_task2, dim=1)
    predicted_lab_task3 = torch.argmax(logits_task3, dim=1)

    # accuracy = (predicted_labels == labels).float().mean()
    # total_eval_accuracy += accuracy.item()

    test_predictions_task1.extend(predicted_lab_task1.detach().cpu().numpy())
    test_predictions_task2.extend(predicted_lab_task2.detach().cpu().numpy())
    test_predictions_task3.extend(predicted_lab_task3.detach().cpu().numpy())
    true_labels_task1.extend(labels_task1.detach().cpu().numpy())
    true_labels_task2.extend(labels_task2.detach().cpu().numpy())
    true_labels_task3.extend(labels_task3.detach().cpu().numpy())



# avg_test_accuracy = total_eval_accuracy / len(test_dataloader)
# logging.info(f'Test accuracy: {avg_test_accuracy:.4f}')


classification_report_output = classification_report(true_labels_task2, test_predictions_task2,zero_division=0,digits=6)

logging.info('Classification Report:')
logging.info(classification_report_output)
