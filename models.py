import torch.nn as nn
import torch.nn.functional as F

class WordAVGModel(nn.Module):
    def __init__(self, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, embedding):
        pooled = F.avg_pool2d(embedding, (embedding.shape[1], 1)).squeeze(1) # [batch size, embedding_dim]
        # 这个就相当于在seq_len维度上做了一个平均
        return self.fc(pooled)
