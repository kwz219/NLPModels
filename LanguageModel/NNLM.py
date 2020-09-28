"""
论文地址: https://www.researchgate.net/publication/221618573_A_Neural_Probabilistic_Language_Model
modified from https://github.com/graykode/nlp-tutorial/blob/master/1-1.NNLM/NNLM.py
NNLM分为两个阶段:
    (1)对词进行特征提取(通过C函数将词映射到词向量)
    (2)进行条件概率的计算(隐藏层)
    额外:原始输入层与输出层有一个直连
对于NNLM存在的疑问:
    1.为什么要将输入层和输出层直接相连(即原文中的Wx)
        将原始输入当作基本特征的扩充,有直连可以减少迭代次数
NNLM的缺点:
    1.需提前设定n_step,无法处理变长输入
"""
import torch
import torch.nn as nn

class NNLM(nn.Module):
    def __init__(self,vocab_size,m,n_hidden,n_step):
        """
        原文中的输出计算公式: y=b+Wx+U.tanh(d+Hx)
                          x=(C(Wt-1),...,C(Wt-n+1))
        :param vocab_size:
        :param m: embedding的维数
        :param n_hidden: 隐藏层的维数
        :param n_step: 考虑多少个词,相当于n-gram中的n
        d：隐藏层的bias , H: 隐藏层的连接
        b:输出层的bias , W: 输入层到输出层的连接, U:输出层的权重

        """
        super(NNLM, self).__init__()
        self.n_step=n_step
        self.m=m
        self.C = nn.Embedding(vocab_size, m)
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, vocab_size, bias=False)
        self.W = nn.Linear(n_step * m, vocab_size, bias=False)
        self.b = nn.Parameter(torch.ones(vocab_size))
    def forward(self,x):
        "input x: [batchsize,n_step]"
        X = self.C(x)  # X : [batch_size, n_step, m]
        X = X.view(-1, self.n_step * self.m)  # [batch_size, n_step * m]
        tanhX = torch.tanh(self.d + self.H(X))  # [batch_size, n_hidden]
        output = self.b + self.W(X) + self.U(tanhX)  # [batch_size, n_vocab_size]
        return output



