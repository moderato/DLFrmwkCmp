import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import h5py
from sys import platform
# %matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10,8)

if platform == "darwin":
    root = "/Users/moderato/Downloads/GTSRB/try"
else:
    root = "/home/zhongyilin/Desktop/GTSRB/try"

data_path = root + "/saved_data"

backends = ["neon_mkl", "neon_cpu", "neon_gpu", "keras_tensorflow", "keras_theano", "keras_cntk"]

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
fig1.suptitle("Training loss versus time", y=0.94)
fig2, ax2 = plt.subplots(2, 3)
fig2.suptitle("Training and validation loss versus batch", y=0.94)
markers = ['o', 'x', 'p', 'v', '+', 's']

for i in range(len(backends)):
    b = backends[i]

    train_cost_batch = pd.DataFrame()
    valid_cost_epoch = pd.DataFrame()
    train_epoch_mark = dict()
    
    f = h5py.File(data_path+"/callback_data_{}.h5".format(b), "r")
    actual_length = f['.']['config'].attrs['total_minibatches']
    
    train_cost_batch['{}_loss'.format(b)] = pd.Series(f['.']['cost']['train'][()]).iloc[0:actual_length] # Training loss per batch
    train_cost_batch['{}_t'.format(b)] = pd.Series(f['.']['time']['train_batch'][()]).cumsum().iloc[0:actual_length] # Training time cumsum per batch
    
    valid_cost_epoch['{}_loss'.format(b)] = pd.Series(f['.']['cost']['loss'][()])
    valid_cost_epoch['{}_t'.format(b)] = pd.Series(f['.']['time']['loss'][()])
    
    tmp = (f['.']['time_markers']['minibatch'][()]-1).astype(int).tolist()
    tmp.pop()
    tmp = [0] + tmp
    train_epoch_mark['{}_mark'.format(b)] = tmp
    ax1.plot(train_cost_batch['{}_t'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], \
             	train_cost_batch['{}_loss'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], marker=markers[i])
    ax1.legend(loc='best')
    ax1.set_ylim((-0.1,4))
    
    ax2[int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
             train_cost_batch['{}_loss'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], label='{}_t'.format(b), marker=markers[0])
    ax2[int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
    		 valid_cost_epoch['{}_loss'.format(b)], label='{}_v'.format(b), marker=markers[1])
    ax2[int(i/3)][i%3].legend(loc='best')
    
    f.close()
    
plt.show()
fig1.savefig(root+"/pics/neon_train_loss_time.png", dpi=fig1.dpi)
fig2.savefig(root+"/pics/neon_train_loss_epoch.png", dpi=fig2.dpi)