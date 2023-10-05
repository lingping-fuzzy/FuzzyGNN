import os
import json
import numpy as np

# this is for ablation
# if __name__ == '__main__':
#
#     dataname = ['DD', 'IMDB-BINARY', 'IMDB-MULTI', 'NCI1',
#                 'OHSU', 'Peking_1', 'PROTEINS', 'SYNTHETIC', 'SYNTHETICnew']
#     types = ['GAT+Transformer', 'GENConv+Transformer']
#     base = 'results\\'
#     res = np.zeros([18, 9]); id = 0; # time, params, avg_acc, acc_max, acc_std, avg_precision, avg_f1, avg_recall, avg_auc
#     for dname in dataname:
#         seed = 0
#         for type in types:
#             name = dname + '-'+ type +'-' + 'GPSModel'
#             file = base + name + '//' + 'agg//test' + '//'+ 'best.json'
#             with open(file) as json_file:
#                 data = json.load(json_file)
#                 print("acc:", data['accuracy'])
#                 res[id, 0] = data['time_epoch']
#                 res[id, 1] = data['params']
#                 res[id, 2] = data['accuracy']
#                 res[id, 3] = data['accuracy_max']
#                 res[id, 4] = data['accuracy_std']
#                 if 'precision' in data.keys():
#                     res[id, 5] = data['precision']
#                 if 'recall' in data.keys():
#                     res[id, 6] = data['recall']
#                 res[id, 7] = data['f1']
#                 res[id, 8] = data['auc']
#                 id = id + 1
#
#     np.savetxt("abla_GPSmodel.csv", res, delimiter=",")

## this is for ffdn
# if __name__ == '__main__':
#
#     dataname = ['DD-None+None-FFDNModel-3fn', 'IMDB-BINARY-None+None-FFDNModel-10fn',
#                 'IMDB-MULTI-None+None-FFDNModel-10fn','NCI1-None+None-FFDNModel-3fn',
#                 'OHSU-None+None-FFDNModel-5fn', 'Peking_1-None+None-FFDNModel-5fn',
#                 'PROTEINS-None+None-FFDNModel-3fn', 'SYNTHETIC-None+None-FFDNModel-10fn',
#                 'SYNTHETICnew-None+None-FFDNModel-10fn']
#
#     base = 'results\\'
#     res = np.zeros([9, 9]); id = 0; # time, params, avg_acc, acc_max, acc_std, avg_precision, avg_f1, avg_recall, avg_auc
#     for dname in dataname:
#         seed = 0
#         file = base + dname + '//' + 'agg//test' + '//' + 'best.json'
#         with open(file) as json_file:
#             data = json.load(json_file)
#             print("acc:", data['accuracy'])
#             res[id, 0] = data['time_epoch']
#             res[id, 1] = data['params']
#             res[id, 2] = data['accuracy']
#             res[id, 3] = data['accuracy_max']
#             res[id, 4] = data['accuracy_std']
#             if 'precision' in data.keys():
#                 res[id, 5] = data['precision']
#             if 'recall' in data.keys():
#                 res[id, 6] = data['recall']
#             res[id, 7] = data['f1']
#             res[id, 8] = data['auc']
#             id = id + 1
#
#     np.savetxt("ffdn_model.csv", res, delimiter=",")


# this is for ablation
if __name__ == '__main__':

    dataname = ['DD', 'IMDB-BINARY', 'IMDB-MULTI', 'NCI1',
                'OHSU', 'Peking_1', 'PROTEINS', 'SYNTHETIC', 'SYNTHETICnew']
    types = ['GAT+Transformer', 'GENConv+Transformer']
    base = 'results\\'
    res = np.zeros([69, 9]); id = 0; # time, params, avg_acc, acc_max, acc_std, avg_precision, avg_f1, avg_recall, avg_auc

    for dname in dataname:
        seed = 0
        for type in types:
            for idi in ['0', '1', '2']:
                for fty in ['', '-mix', '-two']:
                    name = dname + '-'+ type +'-' + idi+fty
                    file = base + name + '//' + 'agg//test' + '//' + 'best.json'
                    if os.path.isfile(file):
                        with open(file) as json_file:
                            data = json.load(json_file)
                            print(name)
                            res[id, 0] = data['time_epoch']
                            res[id, 1] = data['params']
                            res[id, 2] = data['accuracy']
                            res[id, 3] = data['accuracy_max']
                            res[id, 4] = data['accuracy_std']
                            if 'precision' in data.keys():
                                res[id, 5] = data['precision']
                            if 'recall' in data.keys():
                                res[id, 6] = data['recall']
                            res[id, 7] = data['f1']
                            res[id, 8] = data['auc']
                            id = id + 1

    np.savetxt("fuzzymodel_GAT.csv", res, delimiter=",")


