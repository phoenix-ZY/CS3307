import numpy as np
import pandas as pd
from datasets import SentimentDataset
from torch.utils.data import random_split
from torch.utils.data import Dataset,DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
import torch
import torch.nn as nn
from utils import *
from train import *

if __name__ == '__main__':
    eppoch = 0
    for i in range(100):
        eppoch+=1
        train_data = pd.read_csv("data/train_easy_processed.csv", encoding = "ISO-8859-1")
        test_data = pd.read_csv("data/test_easy_processed.csv" ,encoding = "ISO-8859-1")
        dataset = SentimentDataset(train_data,maxlength=150) # 构建数据集，可以传入参数词向量的最大长度:maxlength
        
        # ## 缩小数据集
        # train_size = int(0.99 * len(dataset))   
        # valid_size = len(dataset) - train_size
        # train_dataset, dataset = random_split(dataset, [train_size, valid_size])
        # ##
        
        train_size = int(0.9 * len(dataset))  # 使用90%的数据作为训练集
        valid_size = len(dataset) - train_size

        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        test_dataset = SentimentDataset(test_data,maxlength=150)

        batch_size = 64
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        model = DistilBertForSequenceClassification.from_pretrained("models/distilbert-base-uncased", num_labels=2)
        # for param in model.distilbert.parameters():
        #     param.requires_grad = False
        optimizer = AdamW(model.parameters(), lr=2e-5)
        num_epochs = 5
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        if eppoch == 1:
            bert_train(num_epochs, model, train_dataloader,valid_dataloader, device,optimizer,batch_size, load_path= 'results/based_on_proprecessed_model_all/model_{}'.format(eppoch),save_path = 'results/based_on_proprecessed_model_1_all/model_{}'.format(eppoch))
        else:
            bert_train(num_epochs, model, train_dataloader,valid_dataloader, device,optimizer,batch_size ,load_path = 'results/based_on_proprecessed_model_1_all/model_{}'.format(eppoch - 1), save_path = 'results/based_on_proprecessed_model_1_all/model_{}'.format(eppoch))
        # save_results(test_dataloader,device = torch.device('cpu'),model_name = 'results/model2')
        bert_test(test_dataloader,device, 'results/based_on_proprecessed_model_1_all/model_{}'.format(eppoch))