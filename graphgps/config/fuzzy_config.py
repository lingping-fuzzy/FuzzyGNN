from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


def set_cfg_fuzzy(cfg):
    # perform Fuzzy
    cfg.fuzzy = CN()

    cfg.fuzzy.fuzzy_head = 10
    cfg.fuzzy.fuzzyinvar = 2
    cfg.fuzzy.fuzzynum_mfs = 2
    cfg.fuzzy.fz_rescale = True
    cfg.fuzzy.fzindim_rate = 0.5
    cfg.fuzzy.fzdivision_type = 'random'
    cfg.fuzzy.fz_mix = 'oneorder'
    cfg.fuzzy.fuzzynum = 10
    cfg.fuzzy.fuzzylayer = 3

register_config('cfg_fuzzy', set_cfg_fuzzy)
