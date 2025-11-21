import opt
import torch
import random
from AE import AE
import numpy as np
from opt import args
from GAE import IGAE
from torch import nn
from utils import eva
from torch.optim import Adam
from torch.nn import Parameter
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
import os


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
        print(max(label))
        print(int(max(label)))
        for i in range(int(max(label))+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    # return feat, label, adj, sf
    return feat, label, adj

class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


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



class Pre_model(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,
                 n_input, n_z, n_clusters, v=1.0, n_node=None, device=None):
        super(Pre_model, self).__init__()

        self.ae = AE(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)

        self.ae.load_state_dict(torch.load(args.ae_model_save_path))

        self.gae = IGAE(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            n_input=n_input)

        self.gae.load_state_dict(torch.load(args.gae_model_save_path))

        self.a = Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True)
        self.b = Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True)

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, adj):
        z_ae = self.ae.encoder(x)
        z_igae, z_igae_adj = self.gae.encoder(x, adj)
        z_i = self.a * z_ae + self.b * z_igae
        z_l = torch.spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)
        z_g = torch.mm(s, z_l)
        z_tilde = self.gamma * z_g + z_l
        # x_hat, _mean, _disp, _pi = self.ae.decoder(z_tilde)
        x_hat = self.ae.decoder(z_tilde)
        z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj)
        adj_hat = z_igae_adj + z_hat_adj

        # return x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde, _mean, _disp, _pi
        return x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde



print("network setting…")
print("use cuda: {}".format(opt.args.cuda))
device = torch.device("cuda" if opt.args.cuda else "cpu")

# ae和gae的预训练模型
opt.args.ae_model_save_path = 'scSiameseClu/model_pretrain/{}_ae.pkl'.format(opt.args.name)
opt.args.gae_model_save_path = 'scSiameseClu/model_pretrain/{}_gae.pkl'.format(opt.args.name)
# 设置模型保存路径
opt.args.pre_model_save_path = 'scSiameseClu/model_pretrain/{}_pretrain.pkl'.format(opt.args.name)


# 创建路径
os.makedirs(os.path.dirname(opt.args.ae_model_save_path), exist_ok=True)
os.makedirs(os.path.dirname(opt.args.gae_model_save_path), exist_ok=True)
os.makedirs(os.path.dirname(opt.args.pre_model_save_path), exist_ok=True)



# x, y, adj, sf = load_graph_data(opt.args.name, show_details=True)
x, y, adj = load_graph_data(opt.args.name, show_details=True)
adj = torch.FloatTensor(normalize_adj(adj, self_loop=True, symmetry=False)).to(device)

pca = PCA(n_components=opt.args.n_input)
X_pca = pca.fit_transform(x)

dataset = LoadDataset(X_pca)
# dataset = LoadDataset(x)

data = torch.Tensor(dataset.x).to(device)
label = y

model = Pre_model(ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2, ae_n_enc_3=opt.args.ae_n_enc_3,
                  ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2, ae_n_dec_3=opt.args.ae_n_dec_3,
                  gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2, gae_n_enc_3=opt.args.gae_n_enc_3,
                  gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2, gae_n_dec_3=opt.args.gae_n_dec_3,
                  n_input=opt.args.n_input,
                  n_z=opt.args.n_z,
                  n_clusters=opt.args.n_clusters,
                  v=opt.args.freedom_degree,
                  n_node=data.size()[0],
                  device=device).to(device)


def Pretrain(model, data, adj, label):
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):

        x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde = model(data, adj)

        loss_1 = F.mse_loss(x_hat, data)
        loss_2 = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_3 = F.mse_loss(adj_hat, adj)

        loss_4 = F.mse_loss(z_ae, z_igae)
        loss = loss_1 + args.alpha * loss_2 + args.beta * loss_3 + args.omega * loss_4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.save(model.state_dict(), args.pre_model_save_path)

    print("Finish optimization.")


    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())
    acc, nmi, ari, f1 = eva(label, kmeans.labels_, epoch)

Pretrain(model, data, adj, label)
