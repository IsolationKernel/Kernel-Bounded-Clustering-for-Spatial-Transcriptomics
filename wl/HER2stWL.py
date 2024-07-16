from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
import h5py
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import scipy.io
from utilities import create_labels_seq_cont
from scipy.spatial.distance import cdist
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import warnings

warnings.filterwarnings('ignore')
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
start = time.time()
print(start)


def eval_cluster(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)
    print('acc:' + str(acc) + '   nmi:' + str(nmi) + '  ari:' + str(ari))
    return acc, nmi, ari


def toexcel(filename, firstrow, datalist):
    df = pd.DataFrame(datalist, columns=firstrow)
    df.to_csv(filename, index=False)


def form_str(h, psi, method, acc, nmi, ari):
    strbase = []
    strbase.append(h)
    strbase.append(psi)
    strbase.append(method)
    strbase.append(acc)
    strbase.append(nmi)
    strbase.append(ari)
    return strbase


# 1. threshold is the parameter to set the neighbors of one cell
# 2. in pca the n_components can set the pc number
# 3. in label_sequence, it run the create_labels_seq_cont(), in this function h is the parameter of WL, usually set 7 is enough
if __name__ == '__main__':
    datacell = 'sample_G2'
    data_mat = h5py.File(f'../Dataset/HER2st/{datacell}.h5')
    ## reprocessing h5 file
    x = np.array(data_mat['X'])
    # x =edges
    pos = np.array(data_mat['pos'])
    y = np.array(data_mat['Y'])

    unique_labels = np.unique(y)

    label_to_number = {label: index for index, label in enumerate(unique_labels)}

    labelgd = np.array([label_to_number[label] for label in y])

    data_mat.close()
    pos = pos.T
    # Set coordinates
    # calculate adjacent matrix
    tt_dis = cdist(pos, pos)
    threshold = 6
    neighbor_indices = np.argsort(tt_dis, axis=1)[:, 1:threshold + 1]
    adjacency_matrix = np.zeros((pos.shape[0], pos.shape[0]))
    for i in range(pos.shape[0]):
        adjacency_matrix[i, neighbor_indices[i]] = 1
    adjacent = adjacency_matrix

    # PCA
    pca = PCA(n_components=15)
    pcaData = pca.fit_transform(x)

    '''
    Continuous graph embeddings
    TODO: for package implement a class with same API as for WL
    '''
    node_features = []
    node_features.append(pcaData)
    adj_mat = []
    adj_mat.append(np.asarray(adjacent))
    n_nodes = []
    n_nodes.append(adjacent.shape[0])

    node_features_data = scale(np.concatenate(node_features, axis=0), axis=0)
    splits_idx = np.cumsum(n_nodes).astype(int)
    node_features_split = np.vsplit(node_features_data, splits_idx)
    node_features = node_features_split[:-1]
    labels_sequence = create_labels_seq_cont(node_features, adj_mat, 7)

    newlabellist = []
    newlabellist.append(labelgd)
    labelgd = np.transpose(newlabellist)
    # kmeans = KMeans(n_clusters=cluster_num, random_state=None, copy_x=True, algorithm='auto').fit(labels_sequence[0])
    # acc, nmi, ari = eval_cluster(gd, kmeans.labels_)
    # walktrap
    # print('SpatialPCA walktrap ')
    # walktraplabels = pd.read_csv(dataset_dir + 'spatialPCALatent/hvg_' + datacell + 'spatialpcawalktrap.csv', header=0, index_col=None).values
    # walktraplabels = walktraplabels.flatten().tolist()
    # acc, nmi, ari = eval_cluster(gd, walktraplabels)
    # kmeans = KMeans(n_clusters=7, random_state=None, copy_x=True, algorithm='auto').fit(labels_sequence[0])
    # acc, nmi, ari = eval_cluster(gd, kmeans.labels_)
    matFileName = f'../Dataset/HER2st/wl/{datacell}.mat'
    scipy.io.savemat(matFileName, {'data': labels_sequence[0], 'class': labelgd, 'pos': pos})
end = time.time()
total = end - start
print(total)
