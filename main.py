import numpy as np
import pandas as pd
from datasets import SentimentDataset
from models import WordAVGModel
from torch.utils.data import random_split
from torch.utils.data import Dataset,DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
import torch
from utils import *


train_data = pd.read_csv("data/training.1600000.processed.noemoticon.csv" ,names=["情绪", "编号", "日期", "平台","名称", "评论"],encoding = "ISO-8859-1")
test_data = pd.read_csv("data/testdata.manual.2009.06.14.csv" ,names=["情绪", "编号", "日期", "平台","名称", "评论"],encoding = "ISO-8859-1")
dataset = SentimentDataset(train_data)
train_size = int(0.9999 * len(dataset))  # 使用80%的数据作为训练集
valid_size = len(dataset) - train_size


train_dataset, dataset = random_split(dataset, [train_size, valid_size])

train_size = int(0.9 * len(dataset))  # 使用80%的数据作为训练集
valid_size = len(dataset) - train_size

train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
test_dataset = SentimentDataset(test_data)

batch_size = 2
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

model = DistilBertForSequenceClassification.from_pretrained("models/distilbert-base-uncased", num_labels=2)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    total_loss = 0
    
    # 训练
    model.train()
    for batch in train_dataloader:
        input_ids = batch['sentence'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        model.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss}')

    # 验证
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in valid_dataloader:
            input_ids = batch['sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    valid_accuracy = accuracy_score(y_true, y_pred)
    print(f'Epoch {epoch+1}/{num_epochs} - Valid Accuracy: {valid_accuracy}')

# 保存模型
model.save_pretrained('results/model')


