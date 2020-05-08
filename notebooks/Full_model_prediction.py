import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import mlpredict


gpu = 'V100'
opt = 'SGD'
batchsize = 32

VGG16 = mlpredict.import_tools.import_dnn('VGG16')

VGG16.describe()

batchsize = 2**np.arange(0,6,1)
time_layer = np.zeros([16,6])
time_total = np.zeros(6)

for i in range(len(batchsize)):
    time_total[i], layer, time_layer[:,i] = VGG16.predict(gpu = gpu,
                                                          optimizer = opt,
                                                          batchsize = batchsize[i])
l_unique = layer.copy()
duplicates = [12,11,9,6]
for d in duplicates:
    print(l_unique[d])
    l_unique.pop(d)
    
t_unique = np.delete(time_layer,duplicates,0)

fig,ax = plt.subplots(1,1,figsize=[8,8])
plt.plot(batchsize,t_unique.transpose(),'o-')

ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.yaxis.set_minor_formatter(plt.NullFormatter())

matplotlib.rc('xtick', labelsize=25)
matplotlib.rc('ytick', labelsize=25)

plt.xlabel('batch size',fontsize=25)
plt.ylabel('predicted time (ms)',fontsize=25)

plt.legend(l_unique,bbox_to_anchor=(1.04,1), loc="upper left",fontsize=20)

plt.tight_layout()

plt.show()
