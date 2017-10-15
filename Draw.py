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

data_path = root + "/saved_data/self"

gpu_backends = ["neon", "keras_tensorflow", "keras_theano", "keras_cntk", "mxnet", "pytorch"]
cpu_backends = ["neon", "neon_mkl", "keras_tensorflow", "keras_theano", "keras_cntk", "mxnet", "pytorch"]

# gpu_backends = ["neon", "keras_tensorflow", "keras_theano", "mxnet", "pytorch"]
# cpu_backends = ["neon_mkl", "keras_tensorflow", "keras_theano", "mxnet", "pytorch"]

figs = [None] * 10
axes = [None] * 10

for i in range(0,2):
    b = 'GPU' if i == 0 else 'CPU'
    subplots_num = (2, 3) if b == 'GPU' else (3, 3)
    figs[i*5+0] = plt.figure()
    axes[i*5+0] = figs[i*5+0].add_subplot(1,1,1)
    figs[i*5+0].suptitle("Training loss versus time ({})".format(b), y=0.94)
    figs[i*5+1], axes[i*5+1] = plt.subplots(subplots_num[0], subplots_num[1])
    figs[i*5+1].suptitle("Training and validation loss versus epoch ({})".format(b), y=0.94)

    figs[i*5+2] = plt.figure()
    axes[i*5+2] = figs[i*5+2].add_subplot(1,1,1)
    figs[i*5+2].suptitle("Training accuracy versus time ({})".format(b), y=0.94)
    figs[i*5+3] = plt.figure()
    axes[i*5+3] = figs[i*5+3].add_subplot(1,1,1)
    figs[i*5+3].suptitle("Validation accuracy versus time ({})".format(b), y=0.94)
    figs[i*5+4], axes[i*5+4] = plt.subplots(subplots_num[0], subplots_num[1])
    figs[i*5+4].suptitle("Training and validation accuracy versus epoch ({})".format(b), y=0.94)

markers = ['o', 'x', 'p', 'v', '+', 's', 'P', '*', 'D', 'h', '8']

### GPU ###
for n in range(0, 2):
    backends = gpu_backends if n == 0 else cpu_backends
    device = 'gpu' if n == 0 else 'cpu'
    infer_acc = []
    fastest = None
    for i in range(len(backends)):
        b = backends[i]

        train_cost_batch = pd.DataFrame()
        train_cost_epoch = pd.DataFrame()
        valid_cost_epoch = pd.DataFrame()
        train_acc_batch = pd.DataFrame()
        train_acc_epoch = pd.DataFrame()
        valid_acc_epoch = pd.DataFrame()
        train_epoch_mark = dict()
        
        f = h5py.File(data_path+"/callback_data_{}_{}.h5".format(b, device), "r")
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

        train_cost_epoch['{}_loss'.format(b)] = pd.Series([train_cost_batch['{}_loss'.format(b)][train_epoch_mark['{}_mark'.format(b)][j]:train_epoch_mark['{}_mark'.format(b)][j+1]].mean()\
            if j != (len(train_epoch_mark['{}_mark'.format(b)])-1)\
            else train_cost_batch['{}_loss'.format(b)][train_epoch_mark['{}_mark'.format(b)][j-1]:train_epoch_mark['{}_mark'.format(b)][j]].mean()\
            for j in range(0, len(train_epoch_mark['{}_mark'.format(b)]))
        ])
        train_cost_epoch['{}_t'.format(b)] = pd.Series(train_cost_batch['{}_t'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]].tolist())

        train_acc_epoch['{}_acc'.format(b)] = pd.Series([train_acc_batch['{}_acc'.format(b)][train_epoch_mark['{}_mark'.format(b)][j]:train_epoch_mark['{}_mark'.format(b)][j+1]].mean()\
            if j != (len(train_epoch_mark['{}_mark'.format(b)])-1)\
            else train_acc_batch['{}_acc'.format(b)][train_epoch_mark['{}_mark'.format(b)][j-1]:train_epoch_mark['{}_mark'.format(b)][j]].mean()\
            for j in range(0, len(train_epoch_mark['{}_mark'.format(b)]))
        ])
        train_acc_epoch['{}_t'.format(b)] = pd.Series(train_acc_batch['{}_t'.format(b)].iloc[train_epoch_mark['{}_mark'.format(b)]].tolist())

        # Get the fastest framework
        if fastest is None:
            fastest = (b, train_acc_epoch['{}_t'.format(b)].iloc[-1])
        else:
            if (train_acc_epoch['{}_t'.format(b)].iloc[-1] < fastest[1]):
                fastest = (b, train_acc_epoch['{}_t'.format(b)].iloc[-1])

        # Avg training cost per epoch
        axes[n*5+0].plot(train_cost_epoch['{}_t'.format(b)], train_cost_epoch['{}_loss'.format(b)], marker=markers[i])

        # Avg training cost vs valid cost per epoch
        axes[n*5+1][int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
            train_cost_epoch['{}_loss'.format(b)], label='{}_t'.format(b), marker=markers[0])
        axes[n*5+1][int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
            valid_cost_epoch['{}_loss'.format(b)], label='{}_v'.format(b), marker=markers[1])
        axes[n*5+1][int(i/3)][i%3].legend(loc='best')
        axes[n*5+1][int(i/3)][i%3].yaxis.grid(linestyle='dashdot')
        axes[n*5+1][int(i/3)][i%3].set_xlabel('Epoch')
        axes[n*5+1][int(i/3)][i%3].set_ylabel('Loss')
        
        # Avg training acc per epoch
        axes[n*5+2].plot(train_acc_epoch['{}_t'.format(b)], -1*train_acc_epoch['{}_acc'.format(b)]+101, marker=markers[i])
        
        # Valid acc per epoch
        axes[n*5+3].plot(train_acc_epoch['{}_t'.format(b)], -1*valid_acc_epoch['{}_acc'.format(b)]+101, marker=markers[i])
        
        # Avg training acc vs valid acc per epoch
        axes[n*5+4][int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
            train_acc_epoch['{}_acc'.format(b)], label='{}_t'.format(b), marker=markers[0])
        axes[n*5+4][int(i/3)][i%3].plot(range(len(train_epoch_mark['{}_mark'.format(b)])), \
            valid_acc_epoch['{}_acc'.format(b)], label='{}_v'.format(b), marker=markers[1])
        axes[n*5+4][int(i/3)][i%3].legend(loc='best')
        axes[n*5+4][int(i/3)][i%3].yaxis.grid(linestyle='dashdot')
        axes[n*5+4][int(i/3)][i%3].set_xlabel('Epoch')
        axes[n*5+4][int(i/3)][i%3].set_ylabel('Accuracy (%)')
        
        f.close()

    axes[n*5+0].legend(loc='best')
    axes[n*5+0].set_ylim((-0.1,4))
    axes[n*5+0].yaxis.grid(linestyle='dashdot')
    axes[n*5+0].set_xlabel('Time (s)')
    axes[n*5+0].set_ylabel('Loss')

    handles, labels = axes[n*5+2].get_legend_handles_labels()
    axes[n*5+2].legend(handles, [label + ", infer_acc: {:.2f}%".format(acc)\
        if (acc != max(infer_acc)) else "***" + label + ", infer_acc: {:.2f}%***".format(acc)\
        for label, acc in zip(labels, infer_acc)], loc='best')
    axes[n*5+2].axvline(fastest[1], linestyle='dashed', color='#777777')
    axes[n*5+2].text(fastest[1]+50, 15, fastest[0], size=12)
    axes[n*5+2].set_ylim((0.9,101))
    axes[n*5+2].set_yscale('log')
    axes[n*5+2].set_yticks([101-num for num in (list(range(0,11,1)) + list(range(11,91,10)) + list(range(91,101,1)))])
    axes[n*5+2].set_yticklabels(([''] * 11 + list(range(20,100,10)) + list(range(91,101,1))))
    axes[n*5+2].invert_yaxis()
    axes[n*5+2].yaxis.grid(linestyle='dashdot')
    axes[n*5+2].set_xlabel('Time (s)')
    axes[n*5+2].set_ylabel('Accuracy (%)')

    handles, labels = axes[n*5+3].get_legend_handles_labels()
    axes[n*5+3].legend(handles, [label + ", infer_acc: {:.2f}%".format(acc)\
        if (acc != max(infer_acc)) else "***" + label + ", infer_acc: {:.2f}%***".format(acc)\
        for label, acc in zip(labels, infer_acc)], loc='best')
    axes[n*5+3].axvline(fastest[1], linestyle='dashed', color='#777777')
    axes[n*5+3].text(fastest[1]+50, 15, fastest[0], size=12)
    axes[n*5+3].set_ylim((0.9,101))
    axes[n*5+3].set_yscale('log')
    axes[n*5+3].set_yticks([101-num for num in (list(range(0,11,1)) + list(range(11,91,10)) + list(range(91,101,1)))])
    axes[n*5+3].set_yticklabels(([''] * 11 + list(range(20,100,10)) + list(range(91,101,1))))
    axes[n*5+3].invert_yaxis()
    axes[n*5+3].yaxis.grid(linestyle='dashdot')
    axes[n*5+3].set_xlabel('Time (s)')
    axes[n*5+3].set_ylabel('Accuracy (%)')

    figs[n*5+0].savefig(root+"/pics/{}_train_loss_versus_time.png".format(device), dpi=figs[n*5+0].dpi)
    figs[n*5+1].savefig(root+"/pics/{}_train_and_valid_loss_versus_epoch.png".format(device), dpi=figs[n*5+1].dpi)
    figs[n*5+2].savefig(root+"/pics/{}_train_acc_versus_time.png".format(device), dpi=figs[n*5+2].dpi)
    figs[n*5+3].savefig(root+"/pics/{}_valid_acc_versus_time.png".format(device), dpi=figs[n*5+3].dpi)
    figs[n*5+4].savefig(root+"/pics/{}_train_and_valid_acc_versus_epoch.png".format(device), dpi=figs[n*5+4].dpi)
    
plt.show()