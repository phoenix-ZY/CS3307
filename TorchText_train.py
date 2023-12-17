import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.utils.data import random_split
import time
from transformers import DistilBertTokenizerFast
from datasets import SentimentDataset
from models import WordAVGModel
from train import *
SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
if __name__ == '__main__':

    tokenizer =DistilBertTokenizerFast.from_pretrained("models/distilbert-base-uncased")

    train_data = pd.read_csv("data/training.1600000.processed.noemoticon.csv" ,names=['label', 'id', 'day', 'query', 'user', 'tweets'],encoding = "ISO-8859-1")
    test_data = pd.read_csv("data/testdata.manual.2009.06.14.csv" ,names=['label', 'id', 'day', 'query', 'user', 'tweets'],encoding = "ISO-8859-1")
    dataset = SentimentDataset(train_data) # 构建数据集，可以传入参数词向量的最大长度:maxlength

    ## 缩小数据集
    train_size = int(0.9 * len(dataset))   
    valid_size = len(dataset) - train_size
    train_dataset, dataset = random_split(dataset, [train_size, valid_size])
    ##

    train_size = int(0.9 * len(dataset))  # 使用90%的数据作为训练集
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    test_dataset = SentimentDataset(test_data)

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    INPUT_DIM = 30522
    EMBEDDING_DIM = 100
    OUTPUT_DIM = 1
    PAD_IDX = tokenizer.pad_token_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WordAVGModel(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()   
    model = model.to(device)
    criterion = criterion.to(device)
    N_EPOCHS = 10

    train(N_EPOCHS,model,train_dataloader,valid_dataloader, device, optimizer, criterion,batch_size,save_path="results/wordavgmodel/wordavgmodel.pt")
    model = WordAVGModel(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
    test(test_dataloader,device,model,"results/wordavgmodel/wordavgmodel.pt")