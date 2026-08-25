import os
import torch
import torch.nn as nn


class Algorithm(nn.Module):
    """Base class for DGFraud models."""

    def __init__(self, **kwargs):
        super(Algorithm, self).__init__()
        self.nodes = None
        self.weight_decay = kwargs.get('weight_decay', 0.01)

    def l2_penalty(self):
        return sum(param.pow(2.0).sum() for param in self.parameters())

    def save(self, path='tmp/model.pt'):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), path)
        print('Model saved in file: {}'.format(path))

    def load(self, path='tmp/model.pt'):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print('Model restored from file: {}'.format(path))
