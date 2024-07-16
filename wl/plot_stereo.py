import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import scanpy as sc
import numpy as np
from scipy.io import loadmat
from matplotlib.colors import ListedColormap
import matplotlib.transforms as transforms
import h5py

if __name__ == '__main__':

    m = loadmat(f'../Dataset/Stereo-seq/8_0.5_.mat')
    labels = m['Tclass']
    labels = labels.reshape(-1)

    with h5py.File('../Dataset/Stereo-seq/sterseq_output.h5', "r") as f:
        x = f['normalized_expr'][:]
        location = f['location'][:]
    pos = location.T
    xx = pos[:, 0]
    yy = pos[:, 1]

    unique_labels = np.unique(labels)

    num_classes = len(unique_labels)


    # 创建组合颜色映射
    cmap = cm.get_cmap('tab10', num_classes)
    # idx1 = cmap(4)
    # idx2 = cmap(2)
    # idx3 = cmap(1)
    # idx4 = cmap(7)
    # idx5 = cmap(3)
    # idx6 = cmap(6)
    # idx7 = cmap(0)
    # idx8 = cmap(5)
    #
    #
    # custom_colors = [idx1,idx2,idx3,idx4,idx5,idx6,idx7,idx8]
    # combined_colors = np.vstack((custom_colors))
    # cmap = ListedColormap(combined_colors)
    # custom_colors = [
    #     (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
    #     (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
    #     (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
    #     (0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0),
    #     (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
    #     (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),
    #     (0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0),
    #     (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0)
    # ]
    #
    # # 创建组合颜色映射
    # cmap = ListedColormap(custom_colors)



    plt.figure(figsize=(8, 6))

    ax = plt.gca()
    rotation = transforms.Affine2D().rotate_deg(-90) + ax.transData
    for i, label in enumerate(unique_labels):
        plt.scatter(xx[labels == label], yy[labels == label], color=cmap(i), label=f'{label}', s=3, transform=rotation)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=16, markerscale=6)
    plt.subplots_adjust(left=0.1, right=0.8, bottom=0.05, top=0.9)
    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.title(f'KBC', fontsize=26, fontweight='bold')
    plt.axis('off')
    #plt.savefig('figure/KBC.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()
