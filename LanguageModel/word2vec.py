import torch
import torch.nn as nn
import torch.optim as optim
class Word2vec(nn.Module):
    def __init__(self,vocab_size,emb_size):
        super(Word2vec, self).__init__()
        self.vocab_size=vocab_size
        self.emb_size=emb_size
        "注意,若是采用Linear层代替nn.Embedding，那么不应该使用bias,且输入需转换为one-hot 格式"
        self.inemb=nn.Linear(vocab_size,emb_size,bias=False)
        self.outemb=nn.Linear(emb_size,vocab_size,bias=False)
    def forward(self,x):
        #x:[batch_size,vocab_size]
        X=self.inemb(x)
        output=self.outemb(X)
        return output