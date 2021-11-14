import math
import warnings

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

__all__ = ['LogicInference', 'DLMInferenceBase']


def my_gumbel_softmax(logits, tau=1, hard=False, activate_gumbel=False, gumbel_sigma=1, dim=-1):

    if hard or not activate_gumbel:
        gumbels = torch.zeros(logits.shape, device=logits.device)
    else:
        gumbels = gumbel_sigma * -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    #large numbers stability
    M = gumbels.detach().max(dim)[0]
    gumbels = gumbels - M.unsqueeze_(-1)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class DLMInferenceBase(nn.Module):
    def __init__(self, input_dim, output_dim, front_module, identifier, tnorm='product',
            atoms_per_rule=2, with_dropout=True, with_gumbel=True,
            add_bias=True, independant_noise_per_sample=True,
            fuzzy_or=True, add_negation=True):
        super().__init__()

        dims = [input_dim, output_dim]

        self.tnorm = tnorm  # 'minimum', 'product' (smoother)
        self.atoms_per_rule = atoms_per_rule
        self.with_dropout = with_dropout
        self.with_gumbel = with_gumbel
        self.add_bias = add_bias
        self.independant_noise_per_sample = independant_noise_per_sample
        self.fuzzy_or = fuzzy_or
        self.add_negation = add_negation

        self.tau = 1.
        self.gumbel_sigma = 1.
        self.output_dim = output_dim
        self.identifier = identifier

        self.dims = dims
        if self.add_bias:
            self.dims[0] += 1
        self.weight = Parameter(torch.Tensor(self.dims[-1], self.atoms_per_rule, self.dims[-2]))

        if self.with_dropout:
            self.dropout = nn.Dropout(p=0.1)
        else:
            self.dropout = None

        nn.init.kaiming_normal_(self.weight, nonlinearity='sigmoid')

    def update_tau(self, tau):
        self.tau = tau

    def reset_weights(self):
        nn.init.kaiming_normal_(self.weight, nonlinearity='sigmoid')

    def update_gumbel_noise(self, gumbel_noise):
        self.gumbel_sigma = gumbel_noise

    def update_dropout_prob(self, prob):
        if self.dropout:
            self.dropout.p = prob

    def get_output_dim(self, input_dim):
        return self.output_dim

    def forward(self, input, index_meaning=None):

        f = self.weight.unsqueeze(2)
        if self.independant_noise_per_sample and self.training:
            f = f.repeat(1, 1, input.shape[0], 1)

        for _ in range(len(input.shape) - 2):
            f = f.unsqueeze(-2)

        if self.training:
            activated = my_gumbel_softmax(f, tau=self.tau, activate_gumbel=self.with_gumbel,
                                          hard=False, gumbel_sigma=self.gumbel_sigma)

            #compute entropy before applying dropout to avoid log(0)
            if self.training:
                entropies = -(activated * activated.log()).sum(-1)

            if self.with_dropout:
                activated = self.dropout(activated)
                activated = activated / (activated.sum(-1).unsqueeze(-1) + 1e-5)
        else:
             activated = my_gumbel_softmax(f, tau=self.tau, hard=True)

        f = input

        if self.add_bias:
            f2 = torch.ones((*f.shape[:-1], 1), device=input.device,dtype=f.dtype)
            f = torch.cat((f, f2), -1)

        if self.tnorm == 'product':
            if self.fuzzy_or and self.dims[1] != 1:
                if self.add_negation:
                    negf = 1-f
                    negs = activated.shape[1]//2
                    f1 = (f * activated[:self.dims[1] // 2]).sum(-1)
                    f2 = (f * activated[self.dims[1] // 2:, 0:negs]).sum(-1).unsqueeze(1)
                    f2prime = (negf * activated[self.dims[1] // 2:, negs:]).sum(-1).unsqueeze(1)
                    f2 = torch.cat([f2[:, 0, ...], f2prime[:, 0, ...]], 1)
                    ffand = f1[:f1.shape[0]//2].prod(1)
                    ffor = 1. - (1. - f1[f1.shape[0]//2:]).prod(1)
                    ffand2 = f2[:f2.shape[0]//2].prod(1)
                    ffor2 = 1. - (1. - f2[f2.shape[0]//2:]).prod(1)
                    f = torch.cat([ffand, ffor, ffand2, ffor2])
                else:
                    f = (f * activated).sum(-1)
                    ffand = f[:self.dims[1] // 2].prod(1)
                    ffor = 1. - (1. - f[self.dims[1] // 2:]).prod(1)
                    f = torch.cat([ffand, ffor])
            elif not self.fuzzy_or and self.dims[1] != 1:
                if self.add_negation:
                    negf = 1-f
                    negs = activated.shape[1] // 2
                    f1 = (f * activated[:self.dims[1] // 2]).sum(-1)
                    f2 = (f * activated[self.dims[1] // 2:, 0:negs]).sum(-1).unsqueeze(1)
                    f2prime = (negf * activated[self.dims[1] // 2:, negs:]).sum(-1).unsqueeze(1)
                    f2 = torch.cat([f2[:, 0, ...], f2prime[:, 0, ...]], 1)
                    ffand = f1.prod(1)
                    ffand2 = f2.prod(1)
                    f = torch.cat([ffand, ffand2])
                else:
                    f = (f * activated).sum(-1).prod(1)
            else:
                #if self.training:
                f = (f * activated).sum(-1).prod(1)
                #else:
                #    f=(f * activated.bool()).any(-1).all(1)
                #    f=f.float()

                
        elif self.tnorm == 'minimum':
            if self.fuzzy_or and self.dims[1] != 1:
                if self.add_negation:
                    negf = 1-f
                    f1 = (f * activated[:self.dims[1] // 2]).sum(-1)
                    f2 = (f * activated[self.dims[1] // 2:, 0]).sum(-1).unsqueeze(1)
                    f2prime = (negf * activated[self.dims[1] // 2:, 1]).sum(-1).unsqueeze(1)
                    f2 = torch.cat([f2, f2prime], 1)
                    ffand = f1[:f1.shape[0]//2].min(1)[0]
                    ffor = 1. - (1. - f1[f1.shape[0]//2:]).max(1)[0]
                    ffand2 = f2[:f2.shape[0]//2].min(1)[0]
                    ffor2 = 1. - (1. - f2[f2.shape[0]//2:]).max(1)[0]
                    f = torch.cat([ffand, ffor, ffand2, ffor2])
                else:
                    f = (f * activated).sum(-1)
                    ffand = f[:self.dims[1] // 2].min(1)[0]
                    ffor = f[self.dims[1] // 2:].max(1)[0]
                    f = torch.cat([ffand, ffor])
            elif not self.fuzzy_or and self.dims[1] != 1:
                raise() #TODO
            else:
                f = (f * activated).sum(-1).min(1)[0]

        f = f.permute(*range(1, len(f.shape)), 0)

        pred = (f.detach() > 0.5).float()
        sat = 1 - (f.detach() - pred).abs()

        #never flatten the dict here (or it'll be very slow)
        if self.training:
            return f, {'saturation': sat, 'entropies': entropies}
        return f, {'saturation': sat}
