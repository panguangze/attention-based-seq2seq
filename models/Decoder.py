"""
解码层，使用基于attention的单向gru
没有使用teacher_forcing。
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class Decoder(nn.Module):
    MAX_LENGTH = 10
    def __init__(self,hidden_size,output_size,input_dropout_p = 0.0,dropout_p=0.0, max_length=MAX_LENGTH,attention_used=False):
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_dropout_p = input_dropout_p
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attention_use = attention_used

        self.embedding = nn.Embedding(self.output_size,self.hidden_size)
        self.input_dropout = nn.Dropout(p = input_dropout_p)
        self.gru = nn.GRU(self.hidden_size,self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size,self.output_size)
        self.softmax = nn.LogSoftmax()

        if attention_used:
            self.attention = Attention(self.hidden_size)
    
    def forward_one_step(self,input_var,hidden,encoder_outs):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output,hidden = self.gru(embedded,hidden)
        if self.attention_used:
            output,attn = Attention(output,encoder_outs)
        output = self.softmax(self.linear_out(output.view(-1, self.hidden_size))).view(batch_size, output_size, -1)
        return output,hidden,attn

    def forward(self,inputs,encoder_hidden,encoder_outs):
        #不适用teacher_forcing技巧
        decoder_outs = []
        decoder_input = inputs[:,0].unsuqeeze(1)
        #初始化decoder的hidden_state
        decoder_hidden = init_hidden(encoder_hidden)
        for di in range(self.max_length):
            decoder_out,decoder_hidden,attn= self.forward_one_step(decoder_input,decoder_hidden,encoder_outs)
            decoder_out = decoder_out.suqeeze(1)
            decoder_outs.append(decoder_out)
            decoder_input = decoder_outs[-1].topk(1)[1]
        return decoder_outs,decoder_hidden
        def init_hidden(self,encoder_hidden):
            decoder_hidden = torch.cat([encoder_hidden[0:encoder_hidden.size(0):2],encoder_hidden[1:encoder_hidden.size(0):2]], 2)
        return decoder_hidden
