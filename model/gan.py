import torch
from model import encoder, decoder

class GAN(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(GAN, self).__init__()
        embedding_dim = D_in
        internal_dim = H
        hidden = D_out
        
        self.generator = Generator(embedding_dim, hidden, internal_dim)
        self.discriminator = Discriminator(embedding_dim,internal_dim,hidden)
        
        self.NLLLoss = torch.nn.NLLLoss()
        
        
    def forward(self, x):
        x = self.generator(x)
        x = self.discriminator(x)
        return x
        
        
    def loss(y_pred, targets):
        """
            Loss function according to https://arxiv.org/pdf/1710.04087.pdf
            
            :param y_pred: Per minibatch element, probability distribution over TP (=word embedding native  
                           to target space) and FP (=word embedding produced by generator)
            :param targets: Per minibatch element, binary indication of TP or FP wrt. to ground-truth
            
            :return: Loss for generator and discriminator, respectively
        """
        
        # Loss proportional to discriminator's probability of correctly distinguishing TP and FP
        loss_dis = self.NLLLoss(nn.log(y_pred), targets) # NLLLoss needs log(prob_distribution)
        
        # Loss proportional to discriminator's probability of confusing TP and FP
        targets_inverse = torch.ones(list(targets.shape)[0]) - targets
        loss_gen = self.NLLLoss(nn.log(y_pred), targets_inverse)
        return loss_gen, loss_dis
    
    
class Generator(torch.nn.Module):

    def __init__(self, D_in, H, D_out):
        super(Generator, self).__init__()
        self.D_out = D_out
        
        embedding_dim = D_in
        internal_dim = H
        hidden = D_out

        # Encoders
        self.enc_de = encoder.FeedForwardEncoder(embedding_dim, hidden, internal_dim)
        self.enc_nl = encoder.FeedForwardEncoder(embedding_dim, hidden, internal_dim)
        
        # Decoder
        self.dec_en = decoder.FeedForwardDecoder(internal_dim, hidden, embedding_dim)
        
        
    def forward(self, x, lang):
        out = torch.zeros(list(x.shape)[0], self.D_out)
        
        # Iterate through minibatch and process each word token by its language's respective net
        for i in list(x.shape)[0]:
            if lang[i] == 0:
                encoding = self.enc_de(x[i])
            elif lang[i] == 1:
                encoding = self.enc_nl(x[i])
            # Possibility to add more languages here
            
            out[i] = self.dec_en(encoding)
            
        return out
        
        
class Discriminator(torch.nn.Module):

    def __init__(self, D_in, H, D_out=2):
        super(Discriminator, self).__init__()
        self.w1 = torch.nn.Linear(D_in, H)
        self.w2 = torch.nn.Linear(H, D_out)
        self.activation = torch.nn.functional.relu
        self.softmax = torch.nn.Softmax(dim=1) # Take SM along dimension=1, whereas dim=0 == Batch-size
        
        
    def forward(self, x):
        x = self.w1(x)
        x = self.activation(x)
        x = self.w2(x)
        x = self.softmax(x)      # Create probability distribution over embedding being TP or FP
        return x

