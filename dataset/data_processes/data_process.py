import h5py
import scanpy as sc
from sklearn.calibration import LabelEncoder
from preprocess import *
import numpy as np
import os
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import argparse


Group = ['Meuro_human_Pancreas_cell', 'Tabula_Qs_Lung']

label_dataset_1 = ['Macosko_mouse_retina','Shekhar_mouse_retina_raw_data',
                   'Grace_CITE_CBMC_counts_top2000','Sonya_HumanLiver_counts_top5000']


dataset_all = ['Macosko_mouse_retina','Shekhar_mouse_retina_raw_data',
               'Young_human_kidney_counts','Grace_CITE_CBMC_counts_top2000','Sonya_HumanLiver_counts_top5000',
               'Meuro_human_Pancreas_cell','Tabula_Qs_Lung']





# 构图 
def construct_graph(features, label, method, topk=10):
    num = len(label)
    dist = None
    # Several methods of calculating the similarity relationship between samples i and j (similarity matrix Sij)
    if method == 'heat':
        dist = -0.5 * pairwise_distances(features, metric='manhattan') ** 2
        dist = np.exp(dist)

    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)

    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    elif method == 'p':
        y = features.T - np.mean(features.T)
        features = features - np.mean(features)
        dist = np.dot(features, features.T) / (np.linalg.norm(features) * np.linalg.norm(y))

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    adj = np.zeros_like(dist)
    counter = 0
    for i, v in enumerate(inds):
        for vv in v:
            if vv != i and label[vv] != label[i]:
                adj[i, vv] = 1
                counter += 1
    # Save adjacency matrix as 'adj.npy'
    print('method: {}, error rate: {}'.format(method, counter / (num * topk)))
    return adj


# 读取数据
def data_process(dataset_name,method):
    datapath = os.path.join('scSiameseClu/dataste/', dataset_name)

    if dataset_name in ['Meuro_human_Pancreas_cell', 'Tabula_Qs_Lung']:
        x, y = prepro(datapath+'.h5')
    else:
        data = h5py.File(datapath+'.h5','r')
        x = data['X'][:].astype(np.float64)
        y = data['Y'][:]

    if dataset_name == 'Meuro_human_Pancreas_cell':
        x =  np.round(x).astype(int)
            
    if dataset_name in label_dataset_1:
        y = y-1

    encoder_celltype = LabelEncoder()
    y = encoder_celltype.fit_transform(y)

    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    count_X = x
    adata = normalize_sc(adata, 
                      copy=True, 
                      highly_genes=args.highly_genes, 
                      size_factors=True, 
                      normalize_input=True, 
                      logtrans_input=True)
    
    X = adata.X.astype(np.float32)
    X_raw = adata.raw.X
    sf = adata.obs.size_factors
    Y = np.array(adata.obs["Group"])

    print("dataset_name: {}".format(dataset_name))
    adj = construct_graph(X,Y, method = method, topk=10)

    if os.path.isdir("scSiameseClu/dataset/" + dataset_name) == False:
        os.makedirs("scSiameseClu/dataset/" + dataset_name)

    path = "scSiameseClu/dataset/" + dataset_name + "/" + dataset_name
    np.save(path + '{}_adj.npy'.format(dataset_name), adj)
    np.save(path + '{}_feat.npy'.format(dataset_name), X)
    np.save(path + '{}_label.npy'.format(dataset_name), Y)
    np.save(path + '{}_sf.npy'.format(dataset_name), sf)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DATA_PROCESS', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Meuro_human_Pancreas_cell')
    parser.add_argument('--highly_genes', default = 1000)
    parser.add_argument('--method', type=str, default='heat')  #cos, ncos, heat, p  ------ K-nearest neighbor graph
    args = parser.parse_args()
    data_process(args.name, args.method)
    