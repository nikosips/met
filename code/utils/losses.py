import torch
import torch.nn as nn
import torch.nn.functional as F



class ContrastiveLoss(nn.Module):
    '''Contrastive loss.
    Takes as inputs the embeddings of two samples and a target label == 1 if samples come 
    from the same class or 0 otherwise.
    Credits to https://github.com/adambielski/siamese-triplet
    '''

    def __init__(self, margin):
        
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.eps = 1e-9


    def forward(self, output1, output2, target, size_average=True):

        distances = (output2 - output1).pow(2).sum(1)
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        
        return losses.mean() if size_average else losses.sum()