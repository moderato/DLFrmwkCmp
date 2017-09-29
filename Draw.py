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

gpu_backends = ["neon", "keras_tensorflow", "keras_theano", "keras_cntk", "mxnet", "pytorch"]
cpu_backends = ["neon", "neon_mkl", "keras_tensorflow", "keras_theano", "keras_cntk", "mxnet", "pytorch"]

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
fig1.suptitle("Training loss versus time (GPU)", y=0.94)
fig2, ax2 = plt.subplots(2, 3)
fig2.suptitle("Training and validation loss versus batch (GPU)", y=0.94)

fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
fig3.suptitle("Training loss versus time (CPU)", y=0.94)
fig4, ax4 = plt.subplots(3, 3)
fig4.suptitle("Training and validation loss versus batch (CPU)", y=0.94)

markers = ['o', 'x', 'p', 'v', '+', 's', 'P', '*', 'D', 'h', '8']


### GPU ###
for i in range(len(gpu_backends)):
    b = gpu_backends[i]

    train_cost_batch = pd.DataFrame()
    valid_cost_epoch = pd.DataFrame()
    train_epoch_mark = dict()
    
    f = h5py.File(data_path+"/callback_data_{}_gpu.h5".format(b), "r")
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

### CPU ###
for i in range(len(cpu_backends)):
    b = cpu_backends[i]

    train_cost_batch = pd.DataFrame()
    valid_cost_epoch = pd.DataFrame()
    train_epoch_mark = dict()
    
    f = h5py.File(data_path+"/callback_data_{}_cpu.h5".format(b), "r")
    actual_length = f['.']['config'].attrs['total_minibatches']
    
    train_cost_batch['{}_loss'.format(b)] = pd.Series(f['.']['cost']['train'][()]).iloc[0:actual_length] # Training loss per batch
    train_cost_batch['{}_t'.format(b)] = pd.Series(f['.']['time']['train_batch'][()]).cumsum().iloc[0:actual_length] # Training time cumsum per batch
    
    valid_cost_epoch['{}_loss'.format(b)] = pd.Series(f['.']['cost']['loss'][()])
    valid_cost_epoch['{}_t'.format(b)] = pd.Series(f['.']['time']['loss'][()])
    
    tmp = (f['.']['time_markers']['minibatch'][()]-1).astype(int).tolist()
    tmp.pop()
    tmp = [0] + tmp
    train_epoch_mark['{}_mark'.format(b)] = tmp
    ax3.plot(train_cost_batch['{}_t'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], \
                train_cost_batch['{}_loss'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], marker=markers[i])
    ax3.legend(loc='best')
    ax3.set_ylim((-0.1,4))
    
    ax4[int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
             train_cost_batch['{}_loss'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], label='{}_t'.format(b), marker=markers[0])
    ax4[int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
             valid_cost_epoch['{}_loss'.format(b)], label='{}_v'.format(b), marker=markers[1])
    ax4[int(i/3)][i%3].legend(loc='best')
    
    f.close()
    
plt.show()
fig1.savefig(root+"/pics/gpu_train_loss_versus_time.png", dpi=fig1.dpi)
fig2.savefig(root+"/pics/gpu_train_and_valid_loss_versus_batch.png", dpi=fig2.dpi)
fig3.savefig(root+"/pics/cpu_train_loss_versus_time.png", dpi=fig3.dpi)
fig4.savefig(root+"/pics/cpu_train_and_valid_loss_versus_batch.png", dpi=fig4.dpi)
