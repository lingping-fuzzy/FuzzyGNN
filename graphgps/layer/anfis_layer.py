import torch
import torch.nn as nn
from .anfis import AnfisNet
from .membership import make_gauss_mfs
import numpy as np

class AnfisLayer(nn.Module):
    """Fuzzy layer + downsampling fuzzy layers like a forest
    """

    def __init__(self,
                 fuzzy_head, in_dim, indim_rate, fuzzyinvar = 4, fuzzynum_mfs=3,
                 dim_out = 1, mix_fun='oneorder', rescale= True, division = 'random'):
        super().__init__()

        self.fuzzy_head = fuzzy_head
        self.divison = division
        invar = fuzzyinvar
        num_mfs = fuzzynum_mfs
        inout = int(invar*0.5)
        self.rescale = rescale
        used_feat = in_dim
        if self.rescale == True:
            used_feat = int(indim_rate * in_dim)
            self.fuzzy_ln = nn.Linear(in_dim, used_feat)

        self.dot_dim = int(used_feat/self.fuzzy_head)

        invars = []
        self.FN = nn.ModuleList()
        for i in range(invar):
            sigma = 10 / num_mfs  # how to decides this value
            mulist = torch.linspace(-1, 1, num_mfs).tolist()  # it need good values.
            invars.append(('x{}'.format(i), make_gauss_mfs(sigma, mulist)))
        outvars = ['y{}'.format(i) for i in range(inout)]

        for i in range(self.fuzzy_head):
            if mix_fun == 'oneorder':
                hybrid_fun = False
            elif mix_fun == 'twoorder':
                hybrid_fun = True
            else:
                hybrid_fun = True if np.random.rand() > 0.5 else False

            if self.divison == 'random':
                tree = AnfisNet('Simple classifier', invars, outvars, hybrid=hybrid_fun, used_feature=used_feat,
                                division=self.divison)
            elif self.divison == 'uniform':
                tree = AnfisNet('Simple classifier', invars, outvars, hybrid=hybrid_fun, used_feature=self.dot_dim,
                                division=self.divison)
            else:
                print('wrong division')
            self.FN.append(tree)

        self.fuzzy_ln2 = nn.Linear(self.fuzzy_head*inout, dim_out)


    def forward(self, batch):

        h = batch.x
        if self.rescale == True:
            h = self.fuzzy_ln(h)
        probs = []

        if self.divison == 'random':
            for i, Fuzzn in enumerate(self.FN):
                mu = Fuzzn(h)
                probs.append(mu)  # probs: a list= # heads, nodes*out_dim,
        else:
            for i, Fuzzn in enumerate(self.FN):
                sID = i * self.dot_dim
                tID = (i + 1) * self.dot_dim
                mu = Fuzzn(h[:, sID:tID])
                probs.append(mu)

        probs = torch.cat(probs, dim=1)  # nodes*(out_dim*heads)
        h = self.fuzzy_ln2(probs)

        batch.x = h
        return batch


