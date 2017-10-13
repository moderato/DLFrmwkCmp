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

data_path = root + "/saved_data/small"

gpu_backends = ["neon", "keras_tensorflow", "keras_theano", "keras_cntk", "mxnet", "pytorch"]
cpu_backends = ["neon", "neon_mkl", "keras_tensorflow", "keras_theano", "keras_cntk", "mxnet", "pytorch"]

# gpu_backends = ["neon", "keras_tensorflow", "keras_theano", "mxnet", "pytorch"]
# cpu_backends = ["neon_mkl", "keras_tensorflow", "keras_theano", "mxnet", "pytorch"]

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
fig1.suptitle("Training loss versus time (GPU)", y=0.94)
fig2, ax2 = plt.subplots(2, 3)
fig2.suptitle("Training and validation loss versus epoch (GPU)", y=0.94)

fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
fig3.suptitle("Training loss versus time (CPU)", y=0.94)
fig4, ax4 = plt.subplots(3, 3)
fig4.suptitle("Training and validation loss versus epoch (CPU)", y=0.94)

fig5 = plt.figure()
ax5 = fig5.add_subplot(1,1,1)
fig5.suptitle("Training accuracy versus time (GPU)", y=0.94)
fig6 = plt.figure()
ax6 = fig6.add_subplot(1,1,1)
fig6.suptitle("Validation accuracy versus time (GPU)", y=0.94)
fig7, ax7 = plt.subplots(2, 3)
fig7.suptitle("Training and validation accuracy versus epoch (GPU)", y=0.94)

fig8 = plt.figure()
ax8 = fig8.add_subplot(1,1,1)
fig8.suptitle("Training accuracy versus time (CPU)", y=0.94)
fig9 = plt.figure()
ax9 = fig9.add_subplot(1,1,1)
fig9.suptitle("Validation accuracy versus time (CPU)", y=0.94)
fig10, ax10 = plt.subplots(3, 3)
fig10.suptitle("Training and validation accuracy versus epoch (CPU)", y=0.94)

markers = ['o', 'x', 'p', 'v', '+', 's', 'P', '*', 'D', 'h', '8']

infer_acc = []

### GPU ###
for i in range(len(gpu_backends)):
    b = gpu_backends[i]

    train_cost_batch = pd.DataFrame()
    valid_cost_epoch = pd.DataFrame()
    train_acc_batch = pd.DataFrame()
    valid_acc_epoch = pd.DataFrame()
    train_epoch_mark = dict()
    
    f = h5py.File(data_path+"/callback_data_{}_gpu.h5".format(b), "r")
    actual_length = f['.']['config'].attrs['total_minibatches']
    
    train_cost_batch['{}_loss'.format(b)] = pd.Series(f['.']['cost']['train'][()]).iloc[0:actual_length] # Training loss per batch
    train_cost_batch['{}_t'.format(b)] = pd.Series(f['.']['time']['train_batch'][()]).cumsum().iloc[0:actual_length] # Training time cumsum per batch
    
    valid_cost_epoch['{}_loss'.format(b)] = pd.Series(f['.']['cost']['loss'][()])
    valid_cost_epoch['{}_t'.format(b)] = pd.Series(f['.']['time']['loss'][()])

    train_acc_batch['{}_acc'.format(b)] = pd.Series(f['.']['accuracy']['train'][()]).iloc[0:actual_length] # Training loss per batch
    train_acc_batch['{}_t'.format(b)] = pd.Series(f['.']['time']['train_batch'][()]).cumsum().iloc[0:actual_length] # Training time cumsum per batch
    
    valid_acc_epoch['{}_acc'.format(b)] = pd.Series(f['.']['accuracy']['valid'][()])
    valid_acc_epoch['{}_t'.format(b)] = pd.Series(f['.']['time']['loss'][()])

    infer_acc.append(f['.']['infer_acc']['accuracy'][0])

    tmp = (f['.']['time_markers']['minibatch'][()]-1).astype(int).tolist()
    tmp.pop()
    tmp = [0] + tmp
    train_epoch_mark['{}_mark'.format(b)] = tmp

    ax1.plot(train_cost_batch['{}_t'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], \
             	train_cost_batch['{}_loss'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], marker=markers[i])
    ax1.legend(loc='best')
    ax1.set_ylim((-0.1,6.5))
    
    ax2[int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
             train_cost_batch['{}_loss'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], label='{}_t'.format(b), marker=markers[0])
    ax2[int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
    		 valid_cost_epoch['{}_loss'.format(b)], label='{}_v'.format(b), marker=markers[1])
    ax2[int(i/3)][i%3].legend(loc='best')

    ax5.plot(train_acc_batch['{}_t'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], \
                train_acc_batch['{}_acc'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], marker=markers[i])
    handles, labels = ax5.get_legend_handles_labels()
    ax5.legend(handles, [label + ", infer_acc: {:.2f}%".format(acc) for label, acc in zip(labels, infer_acc)], loc='best')
    ax5.set_ylim((-0.1,103.0))

    ax6.plot(train_acc_batch['{}_t'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], \
                valid_acc_epoch['{}_acc'.format(b)], marker=markers[i])
    handles, labels = ax6.get_legend_handles_labels()
    ax6.legend(handles, [label + ", infer_acc: {:.2f}%".format(acc) for label, acc in zip(labels, infer_acc)], loc='best')
    ax6.set_ylim((-0.1,103.0))

    ax7[int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
             train_acc_batch['{}_acc'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], label='{}_t'.format(b), marker=markers[0])
    ax7[int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
             valid_acc_epoch['{}_acc'.format(b)], label='{}_v'.format(b), marker=markers[1])
    ax7[int(i/3)][i%3].legend(loc='best')
    
    f.close()

infer_acc = []

### CPU ###
for i in range(len(cpu_backends)):
    b = cpu_backends[i]

    train_cost_batch = pd.DataFrame()
    valid_cost_epoch = pd.DataFrame()
    train_acc_batch = pd.DataFrame()
    valid_acc_epoch = pd.DataFrame()
    train_epoch_mark = dict()
    
    f = h5py.File(data_path+"/callback_data_{}_cpu.h5".format(b), "r")
    actual_length = f['.']['config'].attrs['total_minibatches']
    
    train_cost_batch['{}_loss'.format(b)] = pd.Series(f['.']['cost']['train'][()]).iloc[0:actual_length] # Training loss per batch
    train_cost_batch['{}_t'.format(b)] = pd.Series(f['.']['time']['train_batch'][()]).cumsum().iloc[0:actual_length] # Training time cumsum per batch
    
    valid_cost_epoch['{}_loss'.format(b)] = pd.Series(f['.']['cost']['loss'][()])
    valid_cost_epoch['{}_t'.format(b)] = pd.Series(f['.']['time']['loss'][()])

    train_acc_batch['{}_acc'.format(b)] = pd.Series(f['.']['accuracy']['train'][()]).iloc[0:actual_length] # Training loss per batch
    train_acc_batch['{}_t'.format(b)] = pd.Series(f['.']['time']['train_batch'][()]).cumsum().iloc[0:actual_length] # Training time cumsum per batch
    
    valid_acc_epoch['{}_acc'.format(b)] = pd.Series(f['.']['accuracy']['valid'][()])
    valid_acc_epoch['{}_t'.format(b)] = pd.Series(f['.']['time']['loss'][()])
    
    infer_acc.append(f['.']['infer_acc']['accuracy'][0])

    tmp = (f['.']['time_markers']['minibatch'][()]-1).astype(int).tolist()
    tmp.pop()
    tmp = [0] + tmp
    train_epoch_mark['{}_mark'.format(b)] = tmp

    ax3.plot(train_cost_batch['{}_t'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], \
                train_cost_batch['{}_loss'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], marker=markers[i])
    ax3.legend(loc='best')
    ax3.set_ylim((-0.1,6.5))
    
    ax4[int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
             train_cost_batch['{}_loss'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], label='{}_t'.format(b), marker=markers[0])
    ax4[int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
             valid_cost_epoch['{}_loss'.format(b)], label='{}_v'.format(b), marker=markers[1])
    ax4[int(i/3)][i%3].legend(loc='best')

    # Somehow the training acc of the last batch in each epoch is abnormally low on neon_mkl. Use the previous batch accuracy instead.
    ax8.plot(train_acc_batch['{}_t'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], \
                train_acc_batch['{}_acc'.format(b)].iloc[[x - 1 if (x != 0 and b == "neon_mkl") else x for x in train_epoch_mark['{}_mark'.format(b)]]], marker=markers[i])
    handles, labels = ax8.get_legend_handles_labels()
    ax8.legend(handles, [label + ", infer_acc: {:.2f}%".format(acc) for label, acc in zip(labels, infer_acc)], loc='best')
    ax8.set_ylim((-0.1,103.0))

    ax9.plot(train_acc_batch['{}_t'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]], \
                valid_acc_epoch['{}_acc'.format(b)], marker=markers[i])
    handles, labels = ax9.get_legend_handles_labels()
    ax9.legend(handles, [label + ", infer_acc: {:.2f}%".format(acc) for label, acc in zip(labels, infer_acc)], loc='best')
    ax9.set_ylim((-0.1,103.0))

    # Same as above
    ax10[int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
             train_acc_batch['{}_acc'.format(b)].iloc[[x - 1 if (x != 0 and b == "neon_mkl") else x for x in train_epoch_mark['{}_mark'.format(b)]]], label='{}_t'.format(b), marker=markers[0])
    ax10[int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
             valid_acc_epoch['{}_acc'.format(b)], label='{}_v'.format(b), marker=markers[1])
    ax10[int(i/3)][i%3].legend(loc='best')
    
    f.close()
    
plt.show()
fig1.savefig(root+"/pics/gpu_train_loss_versus_time.png", dpi=fig1.dpi)
fig2.savefig(root+"/pics/gpu_train_and_valid_loss_versus_epoch.png", dpi=fig2.dpi)
fig3.savefig(root+"/pics/cpu_train_loss_versus_time.png", dpi=fig3.dpi)
fig4.savefig(root+"/pics/cpu_train_and_valid_loss_versus_epoch.png", dpi=fig4.dpi)
fig5.savefig(root+"/pics/gpu_train_acc_versus_time.png", dpi=fig5.dpi)
fig6.savefig(root+"/pics/gpu_valid_acc_versus_time.png", dpi=fig6.dpi)
fig7.savefig(root+"/pics/gpu_train_and_valid_acc_versus_epoch.png", dpi=fig7.dpi)
fig8.savefig(root+"/pics/cpu_train_acc_versus_time.png", dpi=fig8.dpi)
fig9.savefig(root+"/pics/cpu_valid_acc_versus_time.png", dpi=fig9.dpi)
fig10.savefig(root+"/pics/cpu_train_and_valid_acc_versus_epoch.png", dpi=fig10.dpi)