
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from torch.utils.data import dataloader, Dataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import torch.optim as optim

PATH = "model_state_tuned_bal.pt"



# helper function to get the index list from texts
def get_index_list(series, tokenizer):
    token_lists = []
    for index, value in series.items():
#        print(len(value))
        if len(value) >= 500:
            value1 = value[:510]
            token_list = torch.FloatTensor(tokenizer.encode(value1, add_special_tokens=True)) 
        else:
            token_list = torch.FloatTensor(tokenizer.encode(value, add_special_tokens=True))
#         token_list = torch.LongTensor(tokenizer.convert_tokens_to_ids(value))
        token_lists.append(token_list)
    return token_lists


# load train, validation, and test dataset from tsv file
train_dataset = pd.read_csv("clean_train.csv")
val_dataset = pd.read_csv("clean_val.csv")
test_dataset = pd.read_csv("clean_test.csv")

# initialize bert tokenizer to convert texts to indices
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

df_train = train_dataset.iloc[:, :]
df_val = val_dataset.iloc[:, :]
df_test = test_dataset.iloc[:, :]

#create a sampler to handle imbalance data
sample_count = np.array([143720, 15851])
weight = 1. / sample_count
samples_weight = np.array([weight[t] for t in df_train["IsAbuse"].tolist()])
samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

# read all target lists
train_target = df_train[["threat", "insult", "toxic", "IsAbuse"]].values.tolist()
val_target = df_val[["threat", "insult", "toxic", "IsAbuse"]].values.tolist()
test_target = df_test[["threat", "insult", "toxic", "IsAbuse"]].values.tolist()

# apply get_inded_list on text data
train_index = get_index_list(df_train["comment_text"], tokenizer)
val_index = get_index_list(df_val["comment_text"], tokenizer)
test_index = get_index_list(df_test["comment_text"], tokenizer)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("I am using gpu", device)



# Create dataloader class
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
    
# initialize dataloaders for train, val, test respectively
BATCH_SIZE = 10
max_sent_length=510
train_dataset = CommentDataset(train_index, train_target, max_sent_length)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=train_dataset.comment_collate_func,
                                           sampler = sampler
                                           )

val_dataset = CommentDataset(val_index, val_target, max_sent_length)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=val_dataset.comment_collate_func,
                                           shuffle=False)

test_dataset = CommentDataset(test_index, test_target, max_sent_length)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=test_dataset.comment_collate_func,
                                           shuffle=False)

# initialize bert model
bert = BertModel.from_pretrained("bert_model").to(device)

# add two linear layers on bert
class BertClassifier(nn.Module):
    # initialize layers and bert
    def __init__(self, bert, num_classes, dropout_prob, hidden_size):
        super().__init__()
        self.bert = bert.to(device)
        self.linear_layer = nn.Linear(bert.config.hidden_size, hidden_size).to(device)
        self.dropout = nn.Dropout(dropout_prob).to(device)
        self.non_linearity = nn.ReLU().to(device)
        self.clf = nn.Linear(hidden_size, num_classes).to(device)
    
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
        logits = self.dropout(logits).to(device)
        # pass it through linear layer
        #logits = logits.cuda()
        logits = self.linear_layer(logits) 
        #logits = logits.cuda()
        logits = self.non_linearity(logits)
        
        #logits = logits.cuda()
        #logits = logits.to(device)
        # calculate the probability for each label
        logits = self.clf(logits).to(device)
        #logits = logits.cuda()
        prob = torch.sigmoid(logits).to(device)
        criterion = nn.BCELoss()
        loss = 0
        if labels is not None:
            loss = criterion(prob, labels)
        return loss, prob
#         return outputs
        
    
# initialization new model
# if there is saved pt
# load the parameters
model = BertClassifier(bert, 4, 0.5, 32) 
try:
    model.load_state_dict(torch.load(PATH))
    model.eval()
except:
    pass

# define the criterion
criterion = nn.BCELoss()

def evaluate(model,dataloader, threshold = 0.4):
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

                
#             labels_np = np.array(elem)
#           preds = model(batch_text.type(torch.LongTensor))
            # get probability from the model
            batch_index = batch_index.type(torch.LongTensor).to(device)
            loss, prob = model(batch_index, labels = batch_labels)

            prob = list(prob)
            for label_tensor in batch_labels:
                labels.append(label_tensor.cpu().numpy().tolist())
            for tensor in prob:
                lst = []
                lst_roc = []
                for elem in tensor:

                    lst_roc.append(float(elem))
                    if elem>=threshold:
                        lst.append(1)
                    else:
                        lst.append(0)
                all_preds.append(lst)
                all_preds_roc.append(lst_roc)
        labels = np.array(labels)
        all_preds = np.array(all_preds)
        all_preds_roc = np.array(all_preds_roc)
        for i in range(4):
#             print(i)
#             print(len(labels[:,i]))
#             print(labels[:,i])
#             print(all_preds_roc[:,i])
            roc_auc = roc_auc_score(labels[:,i], all_preds_roc[:,i])
            roc_auc_list.append(roc_auc)

            f1 = f1_score(labels[:,i], all_preds[:,i])
            f1_list.append(f1)

    return f1_list, roc_auc_list

optimizer = optim.Adam(model.parameters(), lr=0.01)
def train(model,dataloader, optimizer):
    # for each training epoch
    model.train()
    train_loss_history_batch = []
    for i, (index_batch, batch_labels) in enumerate(dataloader):
        index_batch = index_batch.type(torch.LongTensor).to(device)
        loss, preds = model(index_batch, labels = batch_labels)
        
        optimizer.step()
        optimizer.zero_grad()
        train_loss_history_batch.append(loss.item())
        
    torch.save(model.state_dict(), PATH)
    print(sum(train_loss_history_batch)/len(train_loss_history_batch))
    print("finish train")
    return train_loss_history_batch



hist = []
NUM_EPOCHS=10
f1_list_val = []
roc_auc_list_val = []
for i in range(NUM_EPOCHS):
    print("in epoch",i)
    hist+=train(model, train_loader, optimizer)
    f1_list, roc_auc_list = evaluate(model, train_loader)
    f1_list_val.append(f1_list)
    roc_auc_list_val.append(roc_auc_list)
    print("this is after one epoch")
    print("f1_list_val",f1_list_val)
    print("roc_auc_list_val",roc_auc_list_val)
