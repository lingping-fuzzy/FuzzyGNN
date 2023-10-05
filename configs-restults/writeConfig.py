from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg)
from yacs.config import CfgNode as CN

class Namespace1:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def initial_cfg(cfg):
    r'''
    This function sets the default config value.
    1) Note that for an experiment, only part of the arguments will be used
    The remaining unused arguments won't affect anything.
    So feel free to register any argument in graphgym.contrib.config
    2) We support *at most* two levels of configs, e.g., cfg.dataset.name

    :return: configuration use by the experiment.
    '''
    if cfg is None:
        return cfg

    # Output directory
    cfg.out_dir = 'results'
    # The metric for selecting the best epoch for each run
    cfg.metric_best = 'auto'

    from yacs.config import CfgNode as CN
    cfg.wandb = CN()
    cfg.dataset = CN()
    cfg.gnn = CN()
    cfg.gt = CN()
    cfg.train = CN()
    cfg.model = CN()
    cfg.optim = CN()
    cfg.fuzzy = CN()


    # Prediction head. Use cfg.dataset.task by default
    cfg.gnn.head = 'default'

    # Number of layers before message passing
    cfg.gnn.layers_pre_mp = 0
    # Number of layers after message passing
    cfg.gnn.layers_post_mp = 0

    # Hidden layer dim. Automatically set if train.auto_match = True
    cfg.gnn.dim_inner = 16
    # Whether use batch norm
    cfg.gnn.batchnorm = True
    # Activation
    cfg.gnn.act = 'relu'
    # Dropout
    cfg.gnn.dropout = 0.0

    # Aggregation type: add, mean, max
    # Note: only for certain layers that explicitly set aggregation type
    # e.g., when cfg.gnn.layer_type = 'generalconv'
    cfg.gnn.agg = 'add'
    # Normalize adj
    cfg.gnn.normalize_adj = False


    # Name of the dataset
    cfg.dataset.name = 'Cora'

    # if PyG: look for it in Pytorch Geometric dataset
    # if NetworkX/nx: load data in NetworkX format
    cfg.dataset.format = 'PyG'

    # Dir to load the dataset. If the dataset is downloaded, this is the
    # Task: node, edge, graph, link_pred
    cfg.dataset.task = 'node'

    # Type of task: classification, regression, classification_binary
    # classification_multi
    cfg.dataset.task_type = 'classification'

    # Transductive / Inductive
    # Graph classification is always inductive
    cfg.dataset.transductive = True

    # Whether random split or use custom split: random / custom
    cfg.dataset.split_mode = 'random'

    # Whether to use an encoder for the node features
    cfg.dataset.node_encoder = False

    # Name of node encoder
    cfg.dataset.node_encoder_name = 'Atom'

    # If add batchnorm after node encoder
    cfg.dataset.node_encoder_bn = True

    # Whether to use an encoder for the edge features
    cfg.dataset.edge_encoder = False

    # Name of edge encoder
    cfg.dataset.edge_encoder_name = 'Bond'

    # If add batchnorm after edge encoder
    cfg.dataset.edge_encoder_bn = True

    # Dimension of the encoded features.
    # For now the node and edge encoding dimensions
    # are the same.

    # Total graph mini-batch size
    cfg.train.batch_size = 16
    # Evaluate model on test data every eval period epochs
    cfg.train.eval_period = 10
    # Save model checkpoint every checkpoint period epochs
    cfg.train.ckpt_period = 100
    cfg.train.mode = 'custom'


    # Model type to use
    cfg.model.type = 'gnn'
    # Loss function: cross_entropy, mse
    cfg.model.loss_fun = 'cross_entropy'
    cfg.model.edge_decoding = 'dot'
    cfg.model.graph_pooling = 'add'
    # ===================================



    # optimizer: sgd, adam
    cfg.optim.optimizer = 'adam'

    # Base learning rate
    cfg.optim.base_lr = 0.01

    # L2 regularization
    cfg.optim.weight_decay = 5e-4
    # Maximal number of epochs
    cfg.optim.max_epoch = 200
    cfg.optim.clip_grad_norm =True

    # WandB group

    # Use wandb or not
    cfg.wandb.use = False
    # Wandb project name, will be created in your team if doesn't exist already
    cfg.wandb.project = "gtblueprint"
    # Positional encodings argument group

    # Type of Graph Transformer layer to use
    cfg.gt.layer_type = 'SANLayer'

    # Number of Transformer layers in the model
    cfg.gt.layers = 3

    # Number of attention heads in the Graph Transformer
    cfg.gt.n_heads = 8

    # Size of the hidden node and edge representation
    cfg.gt.dim_hidden = 64
    # Dropout in feed-forward module.
    cfg.gt.dropout = 0.0

    # Dropout in self-attention.
    cfg.gt.attn_dropout = 0.0

    cfg.gt.layer_norm = False

    cfg.gt.batch_norm = True

    cfg.posenc_LapPE = CN()
    # Use extended positional encodings
    cfg.posenc_LapPE.enable = False

    # Neural-net model type within the PE encoder:
    # 'DeepSet', 'Transformer', 'Linear', 'none', ...
    cfg.posenc_LapPE.model = 'none'

    # Size of Positional Encoding embedding
    cfg.posenc_LapPE.dim_pe = 16

    # Number of layers in PE encoder model
    cfg.posenc_LapPE.layers = 3
    cfg.posenc_LapPE.raw_norm_type = 'none'
    cfg.posenc_LapPE.eigen = CN()

    # The normalization scheme for the graph Laplacian: 'none', 'sym', or 'rw'
    cfg.posenc_LapPE.eigen.laplacian_norm = 'sym'

    # The normalization scheme for the eigen vectors of the Laplacian
    cfg.posenc_LapPE.eigen.eigvec_norm = 'L2'

    # Maximum number of top smallest frequencies & eigenvectors to use
    cfg.posenc_LapPE.eigen.max_freqs = 10

    cfg.fuzzy.fuzzy_head = 10
    cfg.fuzzy.fuzzyinvar = 2
    cfg.fuzzy.fuzzynum_mfs = 2
    cfg.fuzzy.fz_rescale = True
    cfg.fuzzy.fzindim_rate = 0.5
    cfg.fuzzy.fzdivision_type = 'random'
    cfg.fuzzy.fz_mix = 'oneorder'
    cfg.fuzzy.fuzzynum = 10
    cfg.fuzzy.fuzzylayer = 3
    # import torch_geometric.graphgym.register as register
    # from graphgps import  fuzzy_config # noqa, register custom modules
    # # Set user customized cfgs
    # for func in register.config_dict.values():
    #     func(cfg)


def writeConfig(cfg):

    args = Namespace1(cfg_file='configs/small/Tu-DD-FuzzyGPS-2.yaml', repeat=1, mark_done=False, opts=['wandb.use', 'False'])

    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    ## load_cfg(cfg, args)
    return cfg

def changeConfig(cfg, id):
    base = 'D:\\copyfile\\fuzzyConfig\\parametertuning\\gen\\'
    base = 'D:\\copyfile\\fuzzyConfig\\FFDNconfi\\'
    # base = 'D:\\copyfile\\fuzzyConfig\\small\\'
    # cfg_file = base + cfg.dataset.name + '-' + str(cfg.gt.layer_type) + '-'+ cfg.model.type + '.yaml'
    # cfg_file = base+ cfg.dataset.name +'-'+ str(cfg.gt.layer_type)+'-'+str(id)+'.yaml'
    # cfg_file = base + cfg.dataset.name + '-' + str(cfg.gt.layer_type) + '-' + str(id) + '-mix' + '.yaml'
    cfg_file = base + cfg.dataset.name + '-' + str(cfg.gt.layer_type) + '-' + cfg.model.type + id +'.yaml'

    with open(cfg_file, 'w') as f:
        cfg.dump(stream=f)

if __name__ == '__main__':
            cfg = CN()
            initial_cfg(cfg)
            cfg = writeConfig(cfg)

            dnames = ['DD', 'NCI1', 'PROTEINS',  'OHSU', 'Peking_1']
            dnames = ['SYNTHETICnew', 'SYNTHETIC', 'IMDB-BINARY', 'IMDB-MULTI']
            # head, invar, mfs, layer,
            # change all parameters -- DD, NCI1, PROTEINS
            # change of base method
            types = ['-ref','-10fn'] #
            # for generate albation
            for type in range(2):
                for dn in range(4):
                    datasetname = dnames[dn]
                    # cfg.seed = 0
                    cfg.dataset.name = datasetname
                    cfg.wandb.project = datasetname
                    cfg.optim.optimizer = 'adam'
                    cfg.dataset.format = 'PyG-TUDataset'
                    cfg.dataset.split_mode = 'cv-kfold-4'
                    cfg.model.type = 'FFDNModel'  # 'GPSModel'
                    cfg.gt.layer_type = 'None+None'
                    cfg.gt.layers = 2
                    cfg.gt.dim_hidden = 18
                    cfg.gnn.layers_post_mp = 3
                    if type == 0:
                        cfg.fuzzy.fuzzynum = 0
                    else:
                        cfg.fuzzy.fuzzynum = 10
                    changeConfig(cfg, types[type])

# if __name__ == '__main__':
#     cfg = CN()
#     initial_cfg(cfg)
#     cfg = writeConfig(cfg)
#
#     dnames = ['DD', 'NCI1', 'PROTEINS', 'OHSU', 'Peking_1']
#     dnames =[ 'SYNTHETICnew', 'SYNTHETIC', 'IMDB-BINARY', 'IMDB-MULTI']
#     # head, invar, mfs, layer,
#     fuzzpam = [[3, 4, 3, 3], [6, 4, 3, 3], [18, 2, 3, 3], [9, 4, 2, 3], [6, 3, 4, 3]]
#     # change all parameters -- DD, NCI1, PROTEINS
#     # change of base method
#     types = ['GENConv+Transformer', 'None+None', 'GAT+Transformer'] #
#     # for generate albation
#     for type in range(3):
#         for dn in range(4):
#             datasetname = dnames[dn]
#             # cfg.seed = 0
#             cfg.dataset.name = datasetname
#             cfg.wandb.project = datasetname
#             cfg.optim.optimizer = 'adam'
#             cfg.dataset.format = 'PyG-TUDataset'
#             cfg.dataset.split_mode = 'cv-kfold-4'
#             cfg.model.type = 'GPSModel'
#             cfg.gt.layer_type = types[type]
#             cfg.gt.layers = 1
#             cfg.gt.dim_hidden = 18
#             cfg.gnn.layers_post_mp = 1
#             cfg.train.batch_size = 4
#             changeConfig(cfg, type)

# if __name__ == '__main__':
#     cfg = CN()
#     initial_cfg(cfg)
#     cfg = writeConfig(cfg)
#
#     dnames = ['DD', 'NCI1', 'PROTEINS', 'OHSU', 'Peking_1']
#     dnames =[ 'SYNTHETICnew', 'SYNTHETIC', 'IMDB-BINARY', 'IMDB-MULTI']
#     # head, invar, mfs, layer,
#     fuzzpam = [[3, 4, 3, 1], [6, 3, 4, 1]] #, [18, 2, 3, 1], [9, 4, 2, 1], [6, 3, 3, 1]
#     # for generate parametertuning---
#     for id in range(2):
#         for dn in range(4):
#             datasetname = dnames[dn]
#             # cfg.seed = 0
#             cfg.dataset.name = datasetname
#             cfg.wandb.project = datasetname
#             cfg.optim.optimizer = 'adam'
#             cfg.dataset.format = 'PyG-TUDataset'
#             cfg.dataset.split_mode = 'cv-kfold-4'
#             cfg.model.type = 'FuzzyGPSModel'
#             cfg.gt.layer_type = 'GENConv+Transformer' #'GAT+Transformer'  #
#             cfg.gt.layers = 1
#             cfg.gt.dim_hidden = 18
#             cfg.gnn.layers_post_mp = 1
#
#             cfg.fuzzy.fuzzy_head= fuzzpam[id][0]
#             cfg.fuzzy.fuzzyinvar= fuzzpam[id][1]
#             cfg.fuzzy.fuzzynum_mfs= fuzzpam[id][2]
#             cfg.fuzzy.fuzzylayer= fuzzpam[id][3]
#             cfg.fuzzy.fzdivision_type= 'uniform'
#             cfg.fuzzy.fz_mix= 'mix'
#             cfg.fuzzy.fuzzynum= 10 #FFDN use
#             cfg.train.batch_size = 4  # 16-for old
#
#             changeConfig(cfg, id)
#
