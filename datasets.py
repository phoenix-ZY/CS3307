import torch
from torch.utils.data import Dataset,DataLoader
from transformers import DistilBertTokenizerFast
from utils import *
from torchtext.data.utils import get_tokenizer
class SentimentDataset(Dataset):
    def __init__(self,data,maxlength = 0) -> None:
        self.sentences, self.labels = easy_preprocess(data)
        self.tokenizer =DistilBertTokenizerFast.from_pretrained("models/distilbert-base-uncased")
        if(maxlength):
            self.max_length = maxlength
        else:
            self.max_length = self.getmax_length()
        print(self.getmax_length())
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(sentence, padding='max_length',max_length = self.max_length, truncation =True)
        sentence = torch.tensor(encoding['input_ids']).squeeze()
        attention_mask = torch.tensor(encoding['attention_mask']).squeeze()
        return {
            'sentence': sentence,
            'label': torch.tensor(label),
            'attention_mask': attention_mask
        }
    def getmax_length(self):
        maxlength = 0
        for i in range(len(self.sentences)):
            if maxlength < len(self.sentences[i]):
                maxlength = len(self.sentences[i])
        return maxlength





    