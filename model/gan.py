import torch
from model import encoder, decoder


class GAN(torch.nn.Module):

    def __init__(self, D_in, H, D_out, src_languages):
        super(GAN, self).__init__()
        embedding_dim = D_in
        internal_dim = H
        hidden = D_out
        
        self.generator = Generator(embedding_dim, hidden, internal_dim, src_languages)
        self.discriminator = Discriminator(embedding_dim, internal_dim, hidden)
        
        self.NLLLoss = torch.nn.NLLLoss()
        
        
    def forward(self, x):
        raise Warning('Calling GAN DIRECTLY NOT IMPLEMENTED!')
        
        
    def loss(self, y_pred, targets, net):
        """
            Loss function according to https://arxiv.org/pdf/1710.04087.pdf
            
        :param y_pred: Per minibatch element, probability distribution over TP (=1=word embedding native
                       to target space) and FP (=0=word embedding produced by generator)
        :param targets: Per minibatch element, binary indication of TP or FP wrt. to ground-truth
        :param net: Loss to be computed for 'gen' vs 'dis'

        :return: Loss for generator or discriminator
        """

        if net == 'dis':
            # Loss proportional to discriminator's probability of correctly distinguishing TP and FP
            loss = self.NLLLoss(torch.log(y_pred), targets)  # NLLLoss needs log(prob_distribution)
        else:
            # Loss proportional to discriminator's probability of confusing TP and FP
            targets_inverse = torch.ones(list(targets.shape)[0]) - targets
            loss = self.NLLLoss(torch.log(y_pred), targets_inverse)
        return loss
    

class Generator(torch.nn.Module):

    def __init__(self, D_in, H, D_out, languages):
        super(Generator, self).__init__()
        self.D_out = D_out
        
        embedding_dim = D_in
        internal_dim = H
        hidden = D_out

        self.encoders = {}
        for language in languages:
            self.encoders[language] = encoder.FeedForwardEncoder(embedding_dim, hidden, internal_dim)
        
        # Decoder
        self.decoder = decoder.FeedForwardDecoder(internal_dim, hidden, embedding_dim)
        
        
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
        
        
class Discriminator(torch.nn.Module):

    def __init__(self, D_in, H, D_out=2):
        super(Discriminator, self).__init__()
        self.w1 = torch.nn.Linear(D_in, H)
        self.w2 = torch.nn.Linear(H, D_out)
        self.activation = torch.nn.functional.relu
        self.softmax = torch.nn.Softmax(dim=1)  # Take SM along dimension=1, whereas dim=0 == Batch-size
        
        
    def forward(self, x):
        x = self.w1(x)
        x = self.activation(x)
        x = self.w2(x)
        x = self.softmax(x)      # Create probability distribution over embedding being TP or FP
        return x
