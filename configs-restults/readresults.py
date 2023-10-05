import os
import json
import numpy as np

base = 'D:\\copyfile\\COPY\\GTvariantSmall\\new\\'
type1 = ['Tu-DD-FuzzyGPS-para-1', 'Tu-DD-FuzzyGPS-para-1-two']


def getResult1():
    see = ['0', '12']
    based = 'D:\\copyfile\\fuzzyresults\\param\\results-'
    for se in see:
        basedd = based + se +'\\'

        for filefolder in os.listdir(basedd):
            path = basedd + filefolder
            print(filefolder)
            path = path +'//' +'agg//test'+'//'
            file = path + 'best.json'
            if os.path.isfile(file):
                with open(file) as json_file:
                    data = json.load(json_file)
                    print("acc:", data['accuracy'])
            else:
                print("---")

def getResult4():
    see = ['0', '12', '20', '42']

    based = 'D:\\copyfile\\fuzzyresults\\param\\GATtran\\results-'
    based = 'D:\\copyfile\\fuzzyresults\\param\\bat4GAT\\results-'
    res = np.zeros(45)
    num = np.ones(45)
    timeval = np.zeros(45)
    res1 = np.zeros([45, 4])
    for it, se in enumerate(see):
        basedd = based + se +'\\'
        id = 0
        for filefolder in os.listdir(basedd):
            path = basedd + filefolder
            print(filefolder)
            path = path +'//' +'agg//test'+'//'
            file = path + 'best.json'
            if os.path.isfile(file):
                with open(file) as json_file:
                    data = json.load(json_file)
                    print("acc:", data['accuracy'])
                    res[id] = res[id] + data['accuracy']
                    res1[id, it] =  data['accuracy']
                    num[id] = num[id]+1
                    timeval[id] = timeval[id] + data['time_epoch']
            id = id + 1
    # num = max(1, num-1)
    z = num - 1
    print(z)
    z[z == 0] = 1
    res = res/z
    print(res)
    repeat = 9
    for i in range(5):
        print('name start')
        print(z[i*repeat:(i+1)*repeat])
        print(res[i * repeat:(i + 1) * repeat])
        stdval = [np.std(res1[i * repeat + x, :]) for x in range(repeat)]
        print(stdval)
        maxval = [np.max(res1[i * repeat+x, :]) for x in range(repeat) ]
        print( maxval)
        print(timeval[i * repeat:(i + 1) * repeat])


def getResult2():
    see = ['0', '12', '20', '42']

    based = 'D:\\copyfile\\fuzzyresults\\param\\results-'
    res = np.zeros([30, 1])
    for se in see:
        basedd = based + se +'\\'
        id = 0
        for filefolder in os.listdir(basedd):
            path = basedd + filefolder
            print(filefolder)
            path = path +'//' +'agg//test'+'//'
            file = path + 'best.json'

            with open(file) as json_file:
                data = json.load(json_file)
                print("acc:", data['accuracy'])
                res[id, 0] = res[id, 0] + data['accuracy']
            id = id + 1
    res = res/4
    print(res)

def getResult():
    based = 'D:\\copyfile\\fuzzyresults\\results-10\\'
    file = []
    datares = {}

    for filefolder in os.listdir(based):
        model = []
        params = []
        acc_test = []
        path = based + filefolder
        print(filefolder)
        path = path +'//' +'agg//test'+'//'
        file = path + 'best.json'
        with open(file) as json_file:
            data = json.load(json_file)
            print("acc:", data['accuracy'])
        # for filename in os.listdir(path):
        #     if filename[-7:] == "st.json":
        #         file.append(filename)
        #         # print(filename)
        #         with open(os.path.join(path, filename), "r") as f:
        #             lines = f.readlines()
        #
        #             for line in lines:
        #
        #                 if line[:17] == "Total Parameters:":
        #                     params.append(line[-7:-1])


def get_infor_all(DATA_NAME = None):

    if DATA_NAME in ['DD', 'ENZYMES', 'ZINC']: #,  'CYCLES'
        type = type1

    based = base + DATA_NAME + '\\'
    file = []
    datares = {}
    splitdatares = {}
    for Tp in type:

        for id in ['max', 'min', 'mean']:
            if DATA_NAME == 'CYCLES':
                f_dir= based+Tp +id+'/_samples_200/results/'
                if Tp == 'TUs_graph_class_transform':
                    f_dir= based+Tp +'/_samples_200/results/'
            elif  DATA_NAME in ['DD', 'ENZYMES', 'PROTEINS_full' , 'ZINC']:
                f_dir= based+Tp +id+'/results/'
                if Tp == 'TUs_graph_class_transform':
                    f_dir= based+Tp +'/results/'
            elif DATA_NAME in ['WikiCS', 'PROTEINS', 'MUTAG']:
                f_dir= based+Tp +id+'/results/'
                if Tp == (DATA_NAME+'_graph_class_transform'):
                    f_dir= based+Tp +'/results/'

            if not os.path.exists(f_dir):
                continue
            split_test = []
            for filename in os.listdir(f_dir):
                model = []
                params = []
                acc_test = []

                if filename[-4:] == ".txt":
                    file.append(filename)
                    # print(filename)
                    with open(os.path.join(f_dir, filename), "r") as f:
                        lines = f.readlines()

                        for line in lines:

                            if line[:17] == "Total Parameters:":
                                params.append(line[-7:-1])
                            if DATA_NAME in [ 'CYCLES']:
                                if line[:14] == "TEST ACCURACY:":
                                    acc_test.append(line[-8:-1])
                                    split_test.append(line[-8:-1])
                            elif  DATA_NAME in ['DD', 'ENZYMES', 'PROTEINS' , 'WikiCS', 'MUTAG']:
                                if line[:22] == "TEST ACCURACY averaged":
                                    acc_test.append(line[-25:-1])
                                if line[:27] == "All Splits Test Accuracies:":
                                    split_test.append(line[27:-2])
                            elif DATA_NAME == 'ZINC':
                                if line[:8] == "TEST MAE":
                                    acc_test.append(line[-7:-1])

                if Tp == 'TUs_graph_class_transform' or Tp == 'WikiCS_graph_class_transform' or Tp == 'PROTEINS_graph_class_transform' or Tp == 'CYCLES_graph_class_transform' or Tp == 'MUTAG_graph_class_transform':
                    rename = DATA_NAME+'_'+Tp.split('_')[3]+'_'+id
                else:
                    rename = DATA_NAME + '_' + Tp.split('_')[3] + '_' + Tp.split('_')[4]+id
                print(rename, '  para:', params[0], " test-avg-mae:   ", acc_test[0])
                datares[rename] = acc_test[0]
            splitdatares[rename] = split_test
    return datares, splitdatares

def stringTfloat(data):
    fdata = np.empty_like(data)
    for id in range(4):
        fdata[id] = float(data[id])
    return fdata

def getResult3():

    see = ['0', '12', '20', '42']

    based = 'D:\\copyfile\\fuzzyresults\\new albation\\results-'
    res1 = np.zeros([15, 4])
    res = np.zeros([15])
    timeval = np.zeros(45)
    for num, se in enumerate(see):
        basedd = based + se +'\\'
        id = 0
        for filefolder in os.listdir(basedd):
            path = basedd + filefolder
            print(filefolder)
            path = path +'//' +'agg//test'+'//'
            file = path + 'best.json'

            with open(file) as json_file:
                data = json.load(json_file)
                print("acc:", data['accuracy'])
                res1[id, num] =  data['accuracy']
                res[id] = res[id] + data['accuracy']
                timeval[id] = timeval[id] + data['time_epoch']
            id = id + 1
    res = res/4
    print(res)
    repeat = 3
    for i in range(5):
        print('name start')
        # print(res[i * repeat:(i + 1) * repeat], '---max ---', np.max(res1[i * repeat, :]),...,
        # np.max(res1[i * repeat+1, :]), np.max(res1[i * repeat+2, :]) )
        print(res[i * repeat:(i + 1) * repeat])
        print(np.std(res1[i * repeat, :]), np.std(res1[i * repeat + 1, :]), np.std(res1[i * repeat + 2, :]))
        print( np.max(res1[i * repeat, :]), np.max(res1[i * repeat + 1, :]), np.max(res1[i * repeat + 2, :]))
        print(timeval[i * repeat:(i + 1) * repeat])

if __name__ == '__main__':
    # time_require()
    DATAset = ['DD', 'ENZYMES', 'PROTEINS', 'WikiCS', 'CYCLES', 'ZINC', 'MUTAG']
    getResult4()
    print('here is albation')
    getResult3()
    # for key in spdata4.keys():
    #     fdata = stringTfloat(spdata4[key])
    #     print(key, fdata)
    # print(spdata4)
    # dicts = dict0, dict1, dict2, dict3, dict4, dict5
    # dict_names = range(1, len(dicts) + 1)
    #
    # with open("my_data.csv", "w") as output_file:
    #     # output_file.write(header)
    #     for k in dict0.keys():
    #         print_str = "{},".format(k)
    #         print_str += "{},".format(dict0.get(k, None))
    #         print_str += "\n"
    #         output_file.write(print_str)

# DD-GAT- GEN- NONE
# [0.741525  0.7703375 0.7618625]
# NCI1
# [0.7324875 0.7436775 0.66634  ]
# OSHU
# [0.575  0.5625 0.5375]
# PEK
# [0.579545 0.613635 0.613635]
# PROTEN
# [0.72043   0.7383525 0.691755 ]

# name start--# DD-GAT- GEN- NONE
# [0.741525  0.7703375 0.7618625] ---max --- 0.76271 Ellipsis 0.80339 0.79322
# name start
# [0.7324875 0.7436775 0.66634  ] ---max --- 0.73638 Ellipsis 0.75097 0.67412
# name start
# [0.575  0.5625 0.5375] ---max --- 0.65 Ellipsis 0.6 0.65
# name start
# [0.579545 0.613635 0.613635] ---max --- 0.63636 Ellipsis 0.63636 0.63636
# name start
# [0.72043   0.7383525 0.691755 ] ---max --- 0.7491 Ellipsis 0.75269 0.69892

