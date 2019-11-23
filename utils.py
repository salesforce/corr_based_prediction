
import torch.nn as nn
import torch.nn.init as init
import torch
from torch.autograd import Variable
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import zeros_, ones_
import tqdm

class AttackPGD(nn.Module):
    def __init__(self, config):
        super(AttackPGD, self).__init__()
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']

    def forward(self, inputs, targets, basic_net):

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = basic_net(x)
                loss = F.cross_entropy(logits, targets, reduction='sum')
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)

        return basic_net(x)

def pairing_loss(logit1, logit2, stochastic_pairing=False):
    if stochastic_pairing:
        exchanged_idx = np.random.permutation(logit1.shape[0])
        stoc_target_logit2 = logit2[exchanged_idx]
        loss = torch.sum( (stoc_target_logit2-logit1)**2 )/logit1.size()[0]
    else:
        loss = torch.sum( (logit2-logit1)**2 )/logit1.size()[0]
    return loss

def dim_permute(h):
    if len(h.size())>2:
        h=h.permute(1,0,2,3).contiguous()
        h = h.view(h.size(0), -1)
    else:
        h=h.permute(1,0).contiguous()
        h = h.view(h.size(0),-1)
    return h


def compute_l2_norm(h, subtract_mean=False):
    h = dim_permute(h)
    N = (h.size(1))
    if subtract_mean:
        mn =  (h).mean(dim=1, keepdim=True)
        h = h-mn

    l2_norm = (h**2).sum()
    return torch.sqrt(l2_norm)

def correlation_reg(hid, targets, within_class=True, subtract_mean=True):
    norm_fn = compute_l2_norm
    if within_class:
        uniq = np.unique(targets)
        reg_=0
        for u in uniq:
            idx = np.where(targets==u)[0]
        
            norm = norm_fn(hid[idx], subtract_mean=subtract_mean)
            reg_ += (norm)**2
    else:
        norm = norm_fn(hid, subtract_mean=subtract_mean)
        reg_ = (norm)**2
    return reg_



def idx2onehot(idx, n, h=1, w=1):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n).cuda()
    onehot.scatter_(1, idx, 1)
    if h*w>1:
        onehot = onehot.view(idx.size(0), n, 1, 1)
        onehot_tensor = torch.ones(idx.size(0), n, h, w).cuda()
        onehot = onehot_tensor* onehot
    return onehot


def _split_train_val(trainset, val_fraction=0, nsamples=-1):
    if nsamples>-1:
        n_train, n_val = int(nsamples), len(trainset)-int(nsamples) 
    else:    
        n_train = int((1. - val_fraction) * len(trainset))
        n_val =  len(trainset) - n_train
    train_subset, val_subset = torch.utils.data.random_split(trainset, (n_train, n_val))
    return train_subset, val_subset


class add_gaussian_noise():
    def __init__(self, std):
        self.std = std
    def __call__(self,x):
        noise = self.std*torch.randn_like(x)
        return x + noise

