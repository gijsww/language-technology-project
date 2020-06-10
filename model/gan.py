import torch
from model import encoder, decoder


class GAN(torch.nn.Module):

    def __init__(self, D_in, H, D_out, mode='linear', src_languages=None):
        super(GAN, self).__init__()
        
        # Construct generator
        if mode is 'nonlinear':
            print('GAN: Nonlinear mode chosen!')
            self.generator = Generator(D_in=D_in, H=H, D_out=D_in, languages=src_languages)
        
        elif mode is 'linear':
            print('GAN: Linear mode chosen!')
            self.generator = LinearGenerator(D_in=D_in, H=H, D_out=D_in)
        
        # Construct discriminator
        self.discriminator = Discriminator(D_in=D_in, H=H, D_out=D_out)
        
        
    def forward(self, x):
        raise Warning('Calling GAN DIRECTLY NOT IMPLEMENTED!')
        
    

class Generator(torch.nn.Module):

    def __init__(self, D_in, H, D_out, languages):
        super(Generator, self).__init__()
        
        internal_dim = H  # Does not necessarily need to coincide

        self.encoders = {}
        for language in languages:
            self.encoders[language] = encoder.FeedForwardEncoder(D_in=D_in, H=H, D_out=internal_dim)
        
        # Decoder
        self.decoder = decoder.FeedForwardDecoder(D_in=internal_dim, H=H, D_out=D_out)
        
        
    def forward(self, x, batch_language):
        """
            Generator's forward pass.
        :param x: Word embedding vectors from a single language space
        :param batch_language: Indication from which language's embedding space word embedding vectors are provided in x
        :return: Translations of x in target language's embedding space
        """
        x = self.encoders[batch_language](x)  # Feed src-lang's embedding vectors through lang's respective encoder
        x = self.decoder(x)
        return x


class LinearGenerator(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(LinearGenerator, self).__init__()
        self.beta = 0.001  # Default from Conneau et al. (https://github.com/facebookresearch/MUSE/blob/master/unsupervised.py#L41)
        
        self.w1 = torch.nn.Linear(D_in, D_out, bias=False)
        
        
    def forward(self, x):
        x = self.w1(x)
        return x

    def orthogonalize(self):
        """
        Orthogonalize the mapping. (iteratively)
        Source: https://github.com/facebookresearch/MUSE/blob/3159355b93f5c3c4883808ba785ba9d18d7f5e81/src/trainer.py#L181
        """
        W = self.w1.weight.data
        W.copy_((1 + self.beta) * W - self.beta * W.mm(W.transpose(0, 1).mm(W)))

        
class Discriminator(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(Discriminator, self).__init__()
        self.w1 = torch.nn.Linear(D_in, H)
        self.w2 = torch.nn.Linear(H, H)
        self.w3 = torch.nn.Linear(H, H)
        self.w4 = torch.nn.Linear(H, D_out)
        self.activation = torch.nn.functional.relu
        self.softmax = torch.nn.Softmax(dim=1)  # Take SM along dimension=1, whereas dim=0 == Batch-size
        
        
    def forward(self, x):
        x = self.w1(x)
        x = self.activation(x)
        x = self.w2(x)
        x = self.activation(x)
        x = self.w3(x)
        x = self.activation(x)
        x = self.w4(x)
        x = self.softmax(x)      # Create probability distribution over embedding being TP or FP
        return x

