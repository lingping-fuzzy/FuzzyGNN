import torch
import torch.nn as nn
from .anfis import AnfisNet
from .membership import make_gauss_mfs


class FuzzyLayer(nn.Module):
    """Fuzzy layer + downsampling fuzzy layers like a forest
    """

    def __init__(self,
                 fuzzy_head, in_dim, indim_rate, fuzzyinvar = 4, fuzzynum_mfs=3,
                 dim_out = 1, sim_fun=False, downsam= True):
        super().__init__()

        self.fuzzy_head = fuzzy_head

        invar = fuzzyinvar
        num_mfs = fuzzynum_mfs
        inout = int(invar*0.5)
        self.downsam = downsam
        used_feat = in_dim
        if self.downsam == True:
            used_feat = int(indim_rate * in_dim)
            self.fuzzy_ln = nn.Linear(in_dim, used_feat)

        invars = []
        self.FN = nn.ModuleList()
        for i in range(invar):
            sigma = 10 / num_mfs  # how to decides this value
            mulist = torch.linspace(-1, 1, num_mfs).tolist()  # it need good values.
            invars.append(('x{}'.format(i), make_gauss_mfs(sigma, mulist)))
        outvars = ['y{}'.format(i) for i in range(inout)]

        for i in range(self.fuzzy_head):
            tree = AnfisNet('Simple classifier', invars, outvars, hybrid=sim_fun, used_feature=used_feat)
            self.FN.append(tree)

        self.fuzzy_ln2 = nn.Linear(self.fuzzy_head*inout, dim_out)


    def forward(self, batch):

        h = batch.x
        if self.downsam == True:
            h = self.fuzzy_ln(h)
        probs = []
        for i, Fuzzn in enumerate(self.FN):
            mu = Fuzzn(h)
            probs.append(mu)  # probs: a list= # heads, nodes*out_dim,

        probs = torch.cat(probs, dim=1)  # nodes*(out_dim*heads)
        h = self.fuzzy_ln2(probs)

        batch.x = h
        return batch


