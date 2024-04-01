# Importing Libraries

import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from transformers import BertModel, BertConfig
from tqdm import tqdm
import logging
import time
import os

# saving to log files

logging.basicConfig(filename=f'./logs/xlmr(Cod)_{time.asctime().replace(" ","_")}.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger object
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a stream handler to print log messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Loading dataset

Cod = pd.read_csv('/data1/aakash/Codemix/Aakash_02/1Humor_Codemix.csv')
Cod = Cod.dropna()


# max_length
max=0
for s in Cod['Sentence']:
    if len(s.split())>max:
        max=len(s.split())
        print(max)


# Creating custom dataset

class CustomDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        try:
            sentence = str(self.sentences[idx])
            label = self.labels[idx]

            encoding = self.tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            #print(encoding['input_ids'])
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            print(f"Problematic sentence: {self.sentences[idx]}")
            print(f"Problematic label: {self.labels[idx]}")
            raise e


# Model

from transformers import XLMRobertaModel
from transformers import XLMRobertaTokenizer

class CustomXLMRobertaModel(nn.Module):
    def __init__(self, num_labels):
        super(CustomXLMRobertaModel, self).__init__()
        self.xlm_roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        
        # Freeze all layers except the top 2
        for param in self.xlm_roberta.parameters():
            param.requires_grad = False

        # Unfreeze the parameters of the top 2 layers
        for param in self.xlm_roberta.encoder.layer[-2:].parameters():
            param.requires_grad = True

        


        self.linear = nn.Linear(self.xlm_roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        

        # Pooling the hidden states
        pooled_output= hidden_states[:,0,:]
        
        pooled_output=torch.squeeze(pooled_output,dim=1)
        
        logits = self.linear(pooled_output)
        return logits



model =  CustomXLMRobertaModel(num_labels=2)
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

model.to(device)
print("")




# Split dataset into training, validation, and test sets
train_df, remaining_df = train_test_split(Cod, test_size=0.3, random_state=random_seed, stratify=Cod['Tag'])
val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=random_seed)


max_len = 128  # Maximum sequence length

# Create training, validation, and test datasets
train_dataset = CustomDataset(train_df['Sentence'].values, train_df['Tag'].values, tokenizer, max_len)
val_dataset = CustomDataset(val_df['Sentence'].values, val_df['Tag'].values, tokenizer, max_len)
test_dataset = CustomDataset(test_df['Sentence'].values, test_df['Tag'].values, tokenizer, max_len)


train_df.Tag.value_counts()
val_df.Tag.value_counts()


# Creating Dataloader

batch_size = 32
epochs = 50
learning_rate = 2e-5

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)


# Training

weightage= torch.tensor([(1162/(2039+1162))*2, (2039/(2039+1162))*2]).to(device)

criterion = nn.CrossEntropyLoss(weight=weightage)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

train_losses = []
val_losses = []
best_val_loss = float('inf')
best_val_f1_score = 0
patience = 4  # Number of epochs to wait for improvement
counter = 0  # Counter to keep track of epochs without improvement


#best_model_save_path
tempdir = '/data1/aakash/Codemix/Aakash_02/.model/'
best_model_params_path = os.path.join(tempdir, f"best_model_xlmr_Cod.pt")


for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
    

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask=attention_mask)  #([64,2])
        
        loss = criterion(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()



    avg_train_loss = total_loss / len(train_dataloader)

    print(f'Epoch {epoch + 1}/{epochs}')
    logging.info(f'Train loss: {avg_train_loss:.4f}')

    train_losses.append(avg_train_loss)




    # Validation

    val_weightage=torch.tensor([419/(267+419),267/(267+419)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=val_weightage)

    model.eval()
    val_predictions = []
    val_labels = []
    val_loss = 0

    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        predicted_labels = torch.argmax(logits, dim=1)

        val_predictions.extend(predicted_labels.detach().cpu().numpy())
        val_labels.extend(labels.detach().cpu().numpy())
        
        val_loss += criterion(logits, labels).item()
        

    epoch_val_accuracy = accuracy_score(val_labels, val_predictions)
    avg_val_loss = val_loss / len(val_dataloader)
    print(f'Validation accuracy: {epoch_val_accuracy:.4f}')
    logging.info(f'Validation loss: {avg_val_loss:.4f}')

    val_f1_score = f1_score(val_labels, val_predictions, average='macro')

    classification_report_epoch = classification_report(val_labels, val_predictions)
    logging.info(f'Classification Report per Epoch {epoch+1}:')
    logging.info(classification_report_epoch)
    
    val_losses.append(avg_val_loss)

    # Early stopping

    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     torch.save(model.state_dict(), best_model_params_path) # Saving best model
    #     counter = 0
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print(f'Early stopping at epoch {epoch + 1}')
    #         break

    if val_f1_score > best_val_f1_score:
        best_val_f1_score = val_f1_score
        torch.save(model.state_dict(), best_model_params_path)
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break


    scheduler.step()


# Plotting the train and validation loss

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Evaluation

total_eval_accuracy = 0
predictions = []
true_labels = []

best_model = CustomXLMRobertaModel(num_labels=2).to(device)
best_model.load_state_dict(torch.load(best_model_params_path))
best_model.eval()

for batch in test_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        logits = best_model(input_ids, attention_mask=attention_mask)

    predicted_labels = torch.argmax(logits, dim=1)
   
    accuracy = (predicted_labels == labels).float().mean()
    total_eval_accuracy += accuracy.item()

    predictions.extend(predicted_labels.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

avg_test_accuracy = total_eval_accuracy / len(test_dataloader)
logging.info(f'Test accuracy: {avg_test_accuracy:.4f}')


classification_report_output = classification_report(true_labels, predictions)
logging.info('Classification Report:')
logging.info(classification_report_output)

# # Create a DataFrame from the list
# df = pd.DataFrame({'Predicted Labels': predictions})

# # Save the DataFrame to a CSV file
# output_filename = 'predicted_labels_codemix(XLMR_without_Trans).csv'
# df.to_csv(output_filename, index=False)


