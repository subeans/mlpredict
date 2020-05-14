import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings("ignore")
import mlpredict


gpu = 'T4'
opt = 'SGD'

MLP = mlpredict.import_tools.import_dnn('MLP')

MLP.describe()

batchsize = 2**np.arange(0,10,1)
time_layer = np.zeros([2,10])
time_total = np.zeros(10)

for i in range(len(batchsize)):
    time_total[i], layer, time_layer[:,i] = MLP.predict(gpu = gpu,
                                                          optimizer = opt,
                                                          batchsize = batchsize[i])
    print ("time_total = " ,time_total[i])
    print(layer)
    print("time_layer = " ,time_layer[:,i])
    print()
