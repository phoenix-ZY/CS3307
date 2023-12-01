import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
raw_data = pd.read_csv("data/training.1600000.processed.noemoticon.csv" ,names=["情绪", "编号", "日期", "平台","名称", "评论"],encoding = "ISO-8859-1")
test_data = pd.read_csv("data/testdata.manual.2009.06.14.csv" ,names=["情绪", "编号", "日期", "平台","名称", "评论"],encoding = "ISO-8859-1")
X = raw_data['评论']
y_dict = {0: 0,  4: 1}
y = raw_data['情绪'].map(y_dict)
X_train , X_valid, y_train , y_valid = train_test_split(X , y, test_size=0.001,random_state =0)


train_reviews = X_train[0:int(0.9 * len(X_valid))].tolist()
train_labels = y_train[0:int(0.9 * len(y_valid))].tolist()
valid_reviews = X_valid[int(0.9 * len(X_valid)):].tolist()
valid_labels = y_valid[int(0.9 * len(y_valid)):].tolist()

tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert-base-uncased")
inputs = tokenizer(train_reviews,  padding=True, truncation=True)

import torch
class reviewDataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self,idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
train_dataset = reviewDataset(inputs,train_labels)

from transformers import DistilBertForSequenceClassification
model = DistilBertForSequenceClassification.from_pretrained("models/distilbert-base-uncased", num_labels=2)
device =torch.device('cuda' if torch.cuda.is_available() else "cpu")
model.to(device)


from transformers import Trainer,TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,  
    logging_dir='./logs',
    logging_steps=10,
)
trainer = Trainer(
    model=model,
    args=training_args ,
    train_dataset=train_dataset,
)
trainer.train()


def to_check_result(test_encoding):
    input_ids = torch.tensor(test_encoding["input_ids"]).to(device)
    attention_mask = torch.tensor(test_encoding["attention_mask"]).to(device)
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
    y = np.argmax(outputs[0].to("cpu").numpy())
    return y
 
y_pred = []
y_true = test_data['情绪'].map(y_dict).tolist()

for i in test_data['评论']:
    test_encoding1 = tokenizer(i,truncation=True, padding=True)
    # input_ids = torch.tensor(X_valid["input ids"]).to(device)
    # attention_mask = torch.tensor(X_valid["attention mask"]).to(device)
    op = to_check_result(test_encoding1)
    y_pred.append(op)

def accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length."
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct / len(y_true)
    return accuracy

print(accuracy_score(y_true,y_pred))