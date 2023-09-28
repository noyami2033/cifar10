import torch
from torch.autograd import Variable

def get_variable(tensor, **kwargs):
    if torch.cuda.is_available():
        result = Variable(tensor.cuda(), **kwargs)
    else:
        result = Variable(tensor, **kwargs)
    return result