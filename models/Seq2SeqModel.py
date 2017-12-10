import torch.nn as nn

class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder,):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_variable, target_variable=None):
        encoder_outs, encoder_hidden = self.encoder(input_variable)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outs=encoder_outs)
        return result