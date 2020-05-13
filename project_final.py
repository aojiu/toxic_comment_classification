
# !pip install sklearn
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW
from torch.utils.data import dataloader, Dataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import torch.optim as optim

PATH = "untuned_threat.pt"

# load train, validation, and test dataset from tsv file
train_dataset = pd.read_csv("clean_train.csv")
val_dataset = pd.read_csv("clean_val.csv")
test_dataset = pd.read_csv("clean_test.csv")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df_train = train_dataset.iloc[:, :]
#df_train["comment_text"] = df_train["comment_text"].astype(str)
df_val = val_dataset.iloc[:, :]
#df_val["comment_text"] = df_val["comment_text"].astype(str)
df_test = test_dataset.iloc[:, :]
#device = torch.device('cpu')
#if torch.cuda.is_available():
device = torch.device('cuda:0')
#df_train.dropna(subset = ["comment_text"], inplace = True)
#df_val.dropna(subset = ["comment_text"], inplace = True)

    
#create a sampler to handle imbalance data
sample_count = np.array([144277,15294])
# sample_count = np.array([893, 107])
weight = 1. / sample_count
samples_weight = np.array([weight[t] for t in df_train["toxic"].tolist()])
samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))


train_target_insult = df_train["threat"].values.tolist()
val_target_insult = df_val["threat"].values.tolist()
test_target_insult = df_test["threat"].values.tolist()


def get_index_list(series, tokenizer):
    token_lists = []
    for index, value in series.items():
#        value = str(value)
        if len(value) >= 512:
            value = value[:510]
        token_list = torch.FloatTensor(tokenizer.encode(value, add_special_tokens=True))
#         token_list = torch.LongTensor(tokenizer.convert_tokens_to_ids(value))
        token_lists.append(token_list)
    return token_lists



train_index = get_index_list(df_train["comment_text"], tokenizer)
val_index = get_index_list(df_val["comment_text"], tokenizer)
test_index = get_index_list(df_test["comment_text"], tokenizer)


class CommentDataset(Dataset):
    def __init__(self, index_list, target_list, max_sent_length=510):
        """
        @param data_list: list of data tokens(indices)
        @param target_list: list of data targets(labels)

        """
        self.index_list = index_list
        self.target_list = target_list
#         self.max_sent_length = max_sent_length
#         assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
      #length of entries
        return len(self.index_list)
        
    def __getitem__(self, key, max_sent_length=None):
        """
        Triggered when you call dataset[i]
        """

#         if max_sent_length is None:
#             max_sent_length = self.max_sent_length
        #get one input data and its label
#         print(key)
#         print(len(self.index_list))
        index = self.index_list[key][:max_sent_length]
        label = self.target_list[key]
        return [index, label]


    def comment_collate_func(self,batch):
        """
        Customized function for DataLoader that dynamically pads the batch so that all 
        data have the same length
        """ 
        # final result should be 2-d
        index_list = [] # store padded sequences
        label_list = []

        for elem in batch:
            #batch is a list containing tuples

            label_list.append(elem[1])
#             encoded = torch.FloatTensor(tokenizer.encode(elem[0], add_special_tokens=True))
#             index_list.append(encoded)
            index_list.append(elem[0])
            
            
        index_list = pad_sequence(index_list, batch_first=True, padding_value=0)
        """
          # Pad the sequences in your data 
          # if their length is less than max_batch_seq_len
          # or trim the sequences that are longer than self.max_sent_length
          # return padded data_list and label_list
          1. TODO: Your code here 
        """
        
        # print("the length is:",max_batch_seq_len)
        return [torch.Tensor(index_list).type(torch.FloatTensor).to(device), torch.Tensor(label_list).type(torch.FloatTensor).to(device)]

    
# def collate_fn(batch, device: torch.device):
#     index_list, label_list = list(zip(*batch))
#     index_list= pad_sequence(x, batch_first=True, padding_value=0)
#     y = torch.stack(y)
#     return [torch.Tensor(index_list).type(torch.LongTensor), torch.Tensor(label_list).type(torch.LongTensor)]
    
BATCH_SIZE = 10
max_sent_length=510
train_dataset = CommentDataset(train_index, train_target_insult, max_sent_length)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=train_dataset.comment_collate_func,
                                           shuffle=True
                                           #sampler = sampler
                                           )

val_dataset = CommentDataset(val_index, val_target_insult, max_sent_length)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=val_dataset.comment_collate_func,
                                           shuffle=True)

test_dataset = CommentDataset(test_index, test_target_insult, max_sent_length)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=test_dataset.comment_collate_func,
                                           shuffle=True)


bert = BertModel.from_pretrained("bert-base-uncased").to(device)
#bert = BertModel.from_pretrained("bert_model_IsAbuse").to(device)


class BertClassifier(nn.Module):
    # initialize layers and bert
    def __init__(self, bert, num_classes, dropout_prob, hidden_size):
        super().__init__()
        self.bert = bert
        self.linear_layer = nn.Linear(bert.config.hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.non_linearity = nn.ReLU()
        self.clf = nn.Linear(hidden_size, num_classes)
    
    # forward prop
    # to get probability for each class
    # use sigmoid to deal with non-exclusive labels
    # bert -> layer -> ReLu -> Dropout -> Layer -> Sigmoid
    # reference transformers doc: 
    # https://huggingface.co/transformers/v2.2.0/_modules/transformers/modeling_bert.html#BertForSequenceClassification
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        
        outputs = self.bert(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask)
        # get the last layer hidden-state of the first token of the sequence
        logits = outputs[1].to(device)
#         logits = self.dropout(logits)
        # pass it through linear layer
        logits = self.linear_layer(logits)
        logits = self.non_linearity(logits)
        # calculate the probability for each label
        logits = self.clf(logits)
        prob = torch.sigmoid(logits)
        # use argmax to get value and index
#         _, argmax = porb.max(-1)
        criterion = nn.BCELoss()
        loss = 0
        if labels is not None:
            print("prob in forward:", prob)
            print("labels in forward:",labels)
            loss = criterion(prob, labels)
            print(loss)
        return loss, prob
#         return outputs
        
    
# initialization test
model = BertClassifier(bert, 1, 0.5, 32).to(device)    
try:
    model.load_state_dict(torch.load(PATH))
    model.eval()
except:
    pass
# define the criterion
criterion = nn.BCELoss()

# evaluate will return f1 score list for each class
# also returns auc_roc score for each class
def evaluate(model,dataloader, threshold = 0.45, which_label = 1):
    # reset
    f1_list = []
    roc_auc_list = []
    model.eval()
    with torch.no_grad():
        #do not need to compute derivatives for these
        all_acc = []
        all_preds = []
        all_preds_roc = []
        labels = []
        for batch_index, batch_labels in dataloader:
            # get probability from the model
            model.eval()
            loss, prob = model(batch_index.type(torch.LongTensor).to(device), labels = batch_labels)

            prob = list(prob)
            print("prob in eval:", prob)
            for label_tensor in batch_labels:
                labels.append(label_tensor.cpu().numpy().tolist())
            for tensor in prob:

                lst_roc = []
                for elem in tensor:

                    lst_roc.append(float(elem))

                all_preds_roc.append(lst_roc)
        labels = np.array(labels)

        all_preds_roc = np.array(all_preds_roc)

        print("labels in eval:",labels)
        print("preds_roc in eval:",all_preds_roc)
        roc_auc = roc_auc_score(labels, all_preds_roc)
        roc_auc_list.append(roc_auc)

    return roc_auc_list


#optimizer = optim.Adam(model.parameters(), lr=0.00005)
optimizer = AdamW(model.parameters(),lr=2e-5, eps=1e-8)
def train(model,dataloader, optimizer):
    # for each training epoch
    model.train()
    train_loss_history_batch = []
    for i, (index_batch, batch_labels) in enumerate(dataloader):
        optimizer.zero_grad()
        loss, preds = model(index_batch.type(torch.LongTensor).to(device), labels = batch_labels)
        train_loss_history_batch.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_history_batch.append(loss.item())
    torch.save(model.state_dict(), PATH)
    print(train_loss_history_batch)
    return train_loss_history_batch
    # end of one training epoch
    

    
hist = []
NUM_EPOCHS=3
f1_list_val = []
roc_auc_list_val = []
for i in range(NUM_EPOCHS):
    print("in epoch")
    hist+=train(model, train_loader, optimizer)
#     f1_list, roc_auc_list = evaluate(model, train_loader)
    roc_auc_list = evaluate(model, val_loader)
#     f1_list_val += f1_list
    roc_auc_list_val += roc_auc_list
    print("this is after one epoch")
#     print("f1_list_val",f1_list_val)
    print("roc_auc_list_val",roc_auc_list_val)
