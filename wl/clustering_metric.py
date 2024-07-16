from sklearn import metrics
from munkres import Munkres
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def translate(true_label, predict_label):
    # best mapping between true_label and predict label
    l1 = np.unique(true_label)

    numclass1 = l1.shape[0]

    # numclass1 = len(l1)
    l2 = np.unique(predict_label)
    numclass2 = len(l2)
    # numclass2 = len(l2)
    # if numclass1 != numclass2:
    #     print('Class Not equal, Error!!!!',numclass2)
    #     return 0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):


        mps = [i1 for i1, e1 in enumerate(true_label) if e1== c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if predict_label[i1] == c2]

            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost)

    # get the match results
    mapping_predict = np.zeros(len(predict_label))
    # new_predict = np.full(len(self.pred_label),-1)
    translate = []
    for i, c in enumerate(l1):
        # correponding label in l2:
        try:
            # 不能确定正确执行的代码
            c2 = l2[indexes[i][1]]
        except:
            # print('error')
            return predict_label,predict_label
        else:


        # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(predict_label) if elm == c2]
            translate.append(c2)
            mapping_predict[ai] = c
    # new_predict = list(map(int, new_predict.tolist()))
    return translate,mapping_predict



def evaluationClusterModelFromLabel(true_label, predict_label):
    # best mapping between true_label and predict label

    acc = metrics.accuracy_score(true_label, predict_label)
    f1_macro = metrics.f1_score(true_label, predict_label, average='macro')
    precision_macro = metrics.precision_score(true_label, predict_label, average='macro')
    recall_macro = metrics.recall_score(true_label, predict_label, average='macro')
    f1_micro = metrics.f1_score(true_label, predict_label, average='micro')
    precision_micro = metrics.precision_score(true_label, predict_label, average='micro')
    recall_micro = metrics.recall_score(true_label, predict_label, average='micro')

    nmi = metrics.normalized_mutual_info_score(true_label, predict_label)
    adjscore = metrics.adjusted_rand_score(true_label, predict_label)
    return acc, nmi,f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro,adjscore

def plot(X, fig, col, size, true_labels):
    ax = fig.add_subplot(1, 1, 1)
    for i, point in enumerate(X):
        ax.scatter(point[0], point[1], s=size, c=col[true_labels[i]])

def plotClusters(self, tqdm, hidden_emb, true_labels):
    tqdm.write('Start plotting using TSNE...')
    # Doing dimensionality reduction for plotting
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(hidden_emb)
    # Plot figure
    fig = plt.figure()
    self.plot(X_tsne, fig, ['red', 'green', 'blue', 'brown', 'purple', 'yellow', 'pink', 'orange'], 4, true_labels)
    fig.show()
    fig.savefig("plot.png")
    tqdm.write("Finished plotting")
