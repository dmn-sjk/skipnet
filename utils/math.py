from torch import nn
import torch.nn.functional as F


class BatchCrossEntropy(nn.Module):
    def __init__(self):
        super(BatchCrossEntropy, self).__init__()

    def forward(self, x, target):
        logp = F.log_softmax(x)
        target = target.view(-1,1)
        output = - logp.gather(1, target)
        return output