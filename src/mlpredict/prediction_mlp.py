import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import mlpredict


gpu = 'V100'
opt = 'SGD'

Lenet5 = mlpredict.import_tools.import_dnn('Lenet5')

Lenet5.describe()

batchsize = 2**np.arange(0,10,1)
time_layer = np.zeros([5,10])
time_total = np.zeros(10)

for i in range(len(batchsize)):
    time_total[i], layer, time_layer[:,i] = Lenet5.predict(gpu = gpu,
                                                          optimizer = opt,
                                                          batchsize = batchsize[i])
    print ("time_total = " ,time_total[i])
    
