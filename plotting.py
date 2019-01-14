import numpy as np
import matplotlib
matplotlib.use('TkAgg') # todo: remove or change if not working
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


def plot_training(lists, labels, cut_top=None):
    ax = plt.figure().gca()
    plt.clf()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=False))
    for l in lists:
        plt.plot(l)
    plt.ylim(bottom=-.025, top=cut_top)
    plt.legend(labels)
    plt.tight_layout()
    plt.savefig('vis/training{}.png'.format('_cuttop_{}'.format(int(cut_top)) if cut_top else ''))
    #plt.show(block=True)

def plot_attrs(attrs):
    plt.figure()
    for i,attr in enumerate(attrs):
        plt.clf()
        hm = sns.heatmap(attr, cmap='jet', xticklabels=False, yticklabels=False, square=True).get_figure()
        plt.tight_layout()
        hm.savefig('vis/attrs/attr_{}.png'.format(i))
        #plt.show(block=True)

def plot_umatrix(arrays, names):
    plt.figure()
    for name,arr in zip(names, arrays):
        plt.clf()
        hm = sns.heatmap(arr, cmap='binary', xticklabels=False, yticklabels=False, square=True).get_figure()
        plt.tight_layout()
        hm.savefig('vis/umatrix/umatrix_{}.png'.format(name))
        #plt.show(block=True)
        plt.close()

def plot_activation(activations, index_2_class):
    plt.figure()
    plt.clf()
    x, y = np.meshgrid(range(activations.shape[2]), range(activations.shape[1]))
    size = np.max(activations, axis=0) * 20.
    colors = np.argmax(activations, axis=0)
    #labels = np.vectorize(lambda l: str(index_2_class[l]))(colors) # can be used to show legend
    plt.scatter(x, y, s=size, c=colors, cmap='Accent')
    plt.grid(True)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('vis/neurons.png')
    #plt.show(block=True)

# Neural Networks (2-AIN-132/15), FMFI UK BA
# (c) Tomas Kuzma, 2017-2018

def limits(values, gap=0.05):
    x0 = np.min(values)
    x1 = np.max(values)
    xg = (x1 - x0) * gap
    return np.array((x0-xg, x1+xg))

def plot_grid_2d(inputs, weights, i_x=0, i_y=1):
    plt.figure()
    plt.clf()
    plt.scatter(inputs[i_x,:], inputs[i_y,:], s=30, c='#999999', edgecolors=[0.4]*3, alpha=0.5)
    n_rows, n_cols, _ = weights.shape
    for r in range(n_rows):
        plt.plot(weights[r,:,i_x], weights[r,:,i_y], c='#e41a1c')
    for c in range(n_cols):
        plt.plot(weights[:,c,i_x], weights[:,c,i_y], c='#e41a1c')
    plt.xlim(limits(inputs[i_x,:]))
    plt.ylim(limits(inputs[i_y,:]))
    plt.tight_layout()
    plt.savefig('vis/model/dim_{}_{}.png'.format(i_x, i_y))
    #plt.show(block=True)
