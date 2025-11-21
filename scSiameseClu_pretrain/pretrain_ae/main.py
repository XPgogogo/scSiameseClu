import torch
from torch import nn
import random
from AE import AE
import numpy as np
from opt import args
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader


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
    print(dataset_name)
    load_path = "scSiameseClu/dataset/" + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    # sf = np.load(load_path+"_sf.npy", allow_pickle=True)

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
    # return feat, label, adj, sf
    return feat, label, adj


def Pretrain_ae(model, train_loader, device):
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_hat, z_hat = model(x)
            loss = 10 * F.mse_loss(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.save(model.state_dict(), args.model_save_path)
    print("Finish optimization.")


print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")

args.model_save_path = 'scSiameseClu/model_pretrain/{}_ae.pkl'.format(args.name)


class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


x, y, adj = load_graph_data(args.name, show_details=True)

pca = PCA(n_components=args.n_input)
X_pca = pca.fit_transform(x)

dataset = LoadDataset(X_pca)
# dataset = LoadDataset(x)
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)

model = AE(
    ae_n_enc_1=args.ae_n_enc_1,
    ae_n_enc_2=args.ae_n_enc_2,
    ae_n_enc_3=args.ae_n_enc_3,
    ae_n_dec_1=args.ae_n_dec_1,
    ae_n_dec_2=args.ae_n_dec_2,
    ae_n_dec_3=args.ae_n_dec_3,
    n_input=args.n_input,
    n_z=args.n_z).to(device)


Pretrain_ae(model, train_loader, device)
