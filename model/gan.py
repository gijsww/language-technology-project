import torch
from model import encoder, decoder


class GAN(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(GAN, self).__init__()
        
        self.generator = Generator(D_in=D_in, D_out=D_in)
        self.discriminator = Discriminator(D_in=D_in, H=H, D_out=D_out)
        
        
    def forward(self, x):
        raise Warning('Calling GAN DIRECTLY NOT IMPLEMENTED!')
        

class Generator(torch.nn.Module):

    def __init__(self, D_in, D_out):
        super(Generator, self).__init__()

        self.mapping = torch.nn.Linear(D_in, D_out, bias=False)

    def forward(self, x):
        x = self.mapping(x)
        return x
        
        
class Discriminator(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(Discriminator, self).__init__()
        self.w1 = torch.nn.Linear(D_in, H)
        self.w2 = torch.nn.Linear(H, H)
        # self.w3 = torch.nn.Linear(H, H)
        self.w4 = torch.nn.Linear(H, D_out)
        self.activation = torch.nn.functional.relu
        self.sigmoid = torch.nn.Sigmoid()  # Take SM along dimension=1, whereas dim=0 == Batch-size
        
        
    def forward(self, x):
        x = self.w1(x)
        x = self.activation(x)
        x = self.w2(x)
        x = self.activation(x)
        # x = self.w3(x)
        # x = self.activation(x)
        x = self.w4(x)
        x = self.sigmoid(x)      # Create probability distribution over embedding being TP or FP
        return x

