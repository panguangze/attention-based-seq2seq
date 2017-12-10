import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,dim):
        super(Attention,self).__init__()
        self.linear = nn.Linear(dim*2,dim)
    def forward(self,output,encoder_outs):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = encoder_outs.size(1)
        attn = torch.bmm(output, encoder_outs.transpose(1, 2))
        attn = F.softmax(attn.view(-1, input_size)).view(batch_size, -1, input_size)

        mix = torch.bmm(attn, encoder_outs)
        combined = torch.cat((mix, output), dim=2)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn