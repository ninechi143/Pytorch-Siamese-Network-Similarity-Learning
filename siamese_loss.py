import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self , margin = 1):

        super(ContrastiveLoss, self).__init__()

        self.margin = margin

        self.register_buffer('zero', torch.Tensor([0]))


    def forward(self, predictions , targets):

        square_pred = 0.5 * torch.square(predictions)
        margin_square = 0.5 * torch.square(torch.maximum(self.margin - predictions , self.zero))

        return torch.mean((targets) * square_pred + (1 - targets) * margin_square)
