import torch
import torch.nn as nn


def glorot(shape):
    tensor = torch.empty(shape)
    nn.init.xavier_uniform_(tensor)
    return tensor


def zeros(shape):
    return torch.zeros(shape)


def ones(shape):
    return torch.ones(shape)
