import torch
import random
import numpy as np
from opt import args
from GAE import IGAE
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.decomposition import PCA


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup_seed(1)


def load_graph_data(dataset_name, show_details=False):
    """
    load graph data
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :return: the features, labels and adj
    """
    load_path = "scSiameseClu/dataset/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)

    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("edge num:   ", int(adj.sum() / 2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(int(max(label)+1)):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    return feat, label, adj


def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)

    return norm_adj


class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def Pretrain_gae(model, data, adj, gamma_value):
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        z_hat, adj_hat = model(data, adj)
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_a = F.mse_loss(adj_hat, adj)
        loss = loss_w + gamma_value * loss_a
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.save(model.state_dict(), args.model_save_path)
    print("Finish optimization.")


device = torch.device("cuda" if args.cuda else "cpu")
print("use cuda: {}".format(args.cuda))

args.model_save_path = 'scSiameseClu/model_pretrain/{}_gae.pkl'.format(args.name)

x, y, adj = load_graph_data(args.name, show_details=True)
norm_adj = normalize_adj(adj, self_loop=True, symmetry=False)

pca = PCA(n_components=args.n_input)
X_pca = pca.fit_transform(x)

dataset = LoadDataset(X_pca)
# dataset = LoadDataset(x)

data = torch.Tensor(dataset.x).to(device)
adj = torch.Tensor(adj).to(device)


model_gae = IGAE(
    gae_n_enc_1=args.gae_n_enc_1,
    gae_n_enc_2=args.gae_n_enc_2,
    gae_n_enc_3=args.gae_n_enc_3,
    gae_n_dec_1=args.gae_n_dec_1,
    gae_n_dec_2=args.gae_n_dec_2,
    gae_n_dec_3=args.gae_n_dec_3,
    n_input=args.n_input,
).to(device)

Pretrain_gae(model_gae, data, adj, args.gamma_value)
