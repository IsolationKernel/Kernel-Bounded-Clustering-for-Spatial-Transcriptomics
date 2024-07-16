import time

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.preprocessing import scale
import datetime
from utilities import load_continuous_graphs, create_labels_seq_cont, create_labels_seq_cont_alltree,retrieve_graph_filenames
import h5py
import pickle, os, numbers
from preprocess import concat_data, preprocessing_rna, norm_adj, SNN_adj

import numpy as np
import scipy
from scipy.sparse import issparse
import pandas as pd
import scanpy as sc
from scipy import io
from anndata import AnnData
from sklearn.preprocessing import StandardScaler
import sklearn

CHUNK_SIZE = 20000

now = datetime.datetime.now()
import scipy.optimize
import scipy.stats
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
import scanpy as sc
import h5py
# 1. threshold is the parameter to set the neighbors of one cell
# 2. in pca the n_components can set the pc number
# 3. in label_sequence, it run the create_labels_seq_cont(), in this function h is the parameter of WL, usually set 7 is enough

start = time.time()
print(start)

file_path = "../Dataset/SlideseqV2hippocampus/slide_seq_v2_output.h5"

with h5py.File(file_path, "r") as f:
    x = f["normalized_expr"][:]
    location = f["location"][:]

pos = location.T
# edges = pd.read_csv('../Dataset/SlideseqV2hippocampus/slide_seq_v2_normexpr.csv', header=None, index_col=None).values
# edges = np.transpose(edges)
# edges = np.delete(edges, 0, axis=1)


# data_mat = h5py.File("../Dataset/SlideseqV2hippocampus/new_mouse_hippocampus_select_count.h5")
# x = edges
# pos = pd.read_csv('../Dataset/SlideseqV2hippocampus/slide_seq_v2_location.csv', header=None, index_col=None).values
# pos = np.array(data_mat['pos'])
# selectbarcodes =  np.array(data_mat['select_cells'])
# selectgenes =  np.array(data_mat['select_genes'])
# data_mat.close()




# PCA
pca = PCA(n_components=20)
pcaData = pca.fit_transform(x)
tt_dis = cdist(pos, pos)
adjmatrixbool = tt_dis <42
adjacent = np.asarray(adjmatrixbool).astype(int)

node_features = []
node_features.append(pcaData)
adj_mat = []
adj_mat.append(np.asarray(adjacent))
n_nodes = []
n_nodes.append(pcaData.shape[0])

node_features_data = scale(np.concatenate(node_features, axis=0), axis=0)
splits_idx = np.cumsum(n_nodes).astype(int)
node_features_split = np.vsplit(node_features_data, splits_idx)
node_features = node_features_split[:-1]
# Generate the label sequences for h iterations
# labels_sequence = create_labels_seq_cont_alltree(node_features, adj_mat, 7)
labels_sequence =  create_labels_seq_cont(node_features, adj_mat, 3)

matFileName = '../Dataset/SlideseqV2hippocampus/wl/slideseqv2h5pc20.mat'
scipy.io.savemat(matFileName, {'data': labels_sequence[0], 'pos': pos})
end = time.time()
total= end -start
print(total)

