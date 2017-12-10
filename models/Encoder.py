'''
编码层，使用BiGru（双向记忆神经网络）
'''
import torch
import torch.nn as nn
from torch.autograd import Variable

class Encoder(nn.Module):
    """
    input_size:输入大小，跟词典的len一样
    embedded_size:embedd层的size
    hidden_size:Encoder层输出的隐藏状态size
    dropout_p:输入层的dropout的比率
    """
    def __init__(self,input_size,embedded_size,hidden_size,dropout_p=0.0):
        super(Encoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedded_size = embedded_size
        self.dropout_p = dropout_p

        self.embedding_layer = nn.Embedding(input_size,embedded_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.gru = nn.GRU(embedded_size,hidden_size // 2,num_layers = 1,bidirectional=True)
        self.hidden_state = self.init_hidden()
    def forward(self,inputs):
        embedded = self.embedding_layer(inputs)
        embedded = self.dropout(embedded)
        #output = embedded.view(len(input),1,-1)
        #output,self.hidden_state = self.gru(output,self.hidden_state)
        output,self.hidden_state = self.gru(embedded,self.hidden_state)
        return output,self.hidden_state
    
    def init_hidden(self):
        return (Variable(torch.randn(1,1,self.hidden_size // 2)),Variable(torch.randn(1,1,self.hidden_size // 2)))
