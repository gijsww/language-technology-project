import torch

class FeedForwardDecoder(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(FeedForwardDecoder, self).__init__()
        self.w1 = torch.nn.Linear(D_in, H)
        self.w2 = torch.nn.Linear(H, D_out)
        self.activation = torch.nn.functional.relu
        
    def forward(self, x):
        x = self.w1(x)
        x = self.activation(x)
        x = self.w2(x)
        return x


class LinearDecoder(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(FeedForwardDecoder, self).__init__()
        self.w1 = torch.nn.Linear(D_in, D_out)
        
    def forward(self, x):
        x = self.w1(x)
        return x

