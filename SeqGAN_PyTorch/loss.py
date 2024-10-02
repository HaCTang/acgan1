# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
class NLLLoss(nn.Module):
    """Self-Defined NLLLoss Function
        计算总的损失，没有显示明确的损失归一化处理，归一化可以除以(sequence_length * batch_size)
    Args:
        weight: Tensor (num_class, )
    """
    def __init__(self, weight):
        super(NLLLoss, self).__init__()
        self.weight = weight

    def forward(self, prob, target):
        """
        Args:
            prob: (N, C) 
            target : (N, )
        """
        N = target.size(0)
        C = prob.size(1)
        weight = Variable(self.weight).view((1, -1))
        weight = weight.expand(N, C)  # (N, C)
        if prob.is_cuda:
            weight = weight.cuda()
        prob = weight * prob

        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        return -torch.sum(loss)



# import torch
# import torch.nn as nn
# from torch.autograd import Variable

# class NLLLoss(nn.Module):
#     """Self-Defined NLLLoss Function for the Generator"""

#     def __init__(self, weight):
#         super(NLLLoss, self).__init__()
#         self.weight = weight

#     def forward(self, prob, target):
#         """
#         Args:
#             prob: (N, C) - Probability distribution over classes (logits)
#             target: (N, ) - Ground truth class labels
#         """
#         N = target.size(0)
#         C = prob.size(1)
#         weight = Variable(self.weight).view((1, -1))
#         weight = weight.expand(N, C)  # (N, C)
#         if prob.is_cuda:
#             weight = weight.cuda()
#         prob = weight * prob

#         one_hot = torch.zeros((N, C))
#         if prob.is_cuda:
#             one_hot = one_hot.cuda()
#         one_hot.scatter_(1, target.data.view((-1, 1)), 1)
#         one_hot = one_hot.type(torch.ByteTensor)
#         one_hot = Variable(one_hot)
#         if prob.is_cuda:
#             one_hot = one_hot.cuda()
#         loss = torch.masked_select(prob, one_hot)
#         return -torch.sum(loss)


class ACGANLoss(nn.Module):
    """ACGAN Loss combining real/fake and class losses for the Discriminator"""

    def __init__(self, l2_reg_lambda=0.01):
        super(ACGANLoss, self).__init__()
        self.bce_loss = nn.BCELoss()  # Binary Cross Entropy Loss for real/fake classification
        self.ce_loss = nn.CrossEntropyLoss()  # Cross Entropy Loss for class classification
        self.l2_reg_lambda = l2_reg_lambda  # L2 正则化系数

    def forward(self, real_fake_pred, real_fake_target, class_pred, class_target, model):
        """
        Args:
            real_fake_pred: (N, 1) - Predictions for real/fake (discriminator output)
            real_fake_target: (N, 1) - True labels for real/fake (1 for real, 0 for fake)
            class_pred: (N, num_classes) - Predicted class logits
            class_target: (N, ) - Ground truth class labels
            model: The discriminator model, used to access weights for L2 regularization
        """
        real_fake_loss = self.bce_loss(real_fake_pred, real_fake_target)
        class_loss = self.ce_loss(class_pred, class_target)
        
        l2_loss = 0
        for param in model.parameters():
            l2_loss += torch.norm(param, p=2)
        l2_loss = self.l2_reg_lambda * l2_loss
        
        total_loss = real_fake_loss + class_loss + l2_loss      
        return total_loss


# Example usage in training loop
def generator_loss(fake_pred, real_class_pred, real_class_target):
    """
    Generator's loss combines NLLLoss (negative log-likelihood) and class prediction loss.
    
    Args:
        fake_pred: (N, num_classes) - Predictions for generated samples
        real_class_pred: (N, num_classes) - Predicted classes from discriminator
        real_class_target: (N, ) - True class labels for the generator
    """
    # Negative Log-Likelihood Loss for Generator
    nll_loss = NLLLoss(weight=torch.ones(fake_pred.size(1)))
    gen_loss = nll_loss(fake_pred, real_class_target)
    
    return gen_loss


def discriminator_loss(real_fake_pred, real_fake_target, class_pred, class_target):
    """
    Discriminator loss combines real/fake classification loss and class prediction loss.
    
    Args:
        real_fake_pred: (N, 1) - Predictions for real/fake classification
        real_fake_target: (N, 1) - Ground truth for real/fake classification
        class_pred: (N, num_classes) - Predictions for class classification
        class_target: (N, ) - Ground truth class labels
    """
    acgan_loss = ACGANLoss()
    disc_loss = acgan_loss(real_fake_pred, real_fake_target, class_pred, class_target)
    
    return disc_loss