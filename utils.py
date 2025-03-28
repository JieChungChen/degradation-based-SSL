import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400
plt.rcParams["xtick.direction"] = 'in'
plt.rcParams["ytick.direction"] = 'in'
params = {
   'axes.labelsize': 16,
   'font.size': 16,
   'legend.fontsize': 16,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16,
   'text.usetex': False,
   'figure.figsize': [6, 4],
   'mathtext.fontset': 'stix',
   'font.family': 'STIXGeneral'
   }
plt.rcParams.update(params)


def find_nearest(array, v_list):
        array = np.asarray(array)
        pos = []
        for v in v_list:
            if np.abs(array - v).min()<5:
                pos.append(np.abs(array - v).argmin())
        return pos


def degradation_loss(seq):
    exp_diff = np.exp(-np.diff(seq/0.1))
    dif = np.mean(-np.log(exp_diff/(exp_diff+1)))
    return dif


def loss_profile(dif):
    """
    plot loss v.s. epoch curve
    """
    plt.plot(dif, c='red', label='degradation_loss', ls='--')
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.yscale('log')
    plt.legend()
    plt.savefig('loss_profile.png')
    plt.close()


def capacity_distribution():
    cell_ids = ['G1', 'V5', 'W4', 'W5', 'W8', 'W9', 'V4', 'W10']
    cm = plt.colormaps['tab10']
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0.0)
    for i, cell in enumerate(cell_ids):
        real_cap = np.load('Stanford_Dataset/capacity_each_cell/%s_capacity.npy'%(cell))
        plt.plot(real_cap[0, :], real_cap[1, :], c=cm(i/7), marker='o', ms=6, alpha=.9, label=cell_ids[i])
    plt.xlabel('Cycle')
    plt.ylabel('Capacity (Ah)')
    frame = plt.legend(fontsize=12, bbox_to_anchor=(1.3, 1)).get_frame()
    frame.set_edgecolor('0')
    plt.savefig('capacity_each_cell.pdf', bbox_inches='tight')
    plt.close()