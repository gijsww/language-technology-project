import torch

class FeedForwardEncoder(torch.nn.Module):
    
    def __init__(self, D_in, H, D_out):
        super(FeedForwardEncoder, self).__init__()
        self.w1 = torch.nn.Linear(D_in, H)
        self.w2 = torch.nn.Linear(H, H)
        self.w3 = torch.nn.Linear(H, H)
        self.w4 = torch.nn.Linear(H, D_out)
        self.activation = torch.nn.functional.relu
        
    def forward(self, x):
        x = self.w1(x)
        x = self.activation(x)
        x = self.w2(x)
        x = self.activation(x)
        x = self.w3(x)
        x = self.activation(x)
        x = self.w4(x)
        return x

