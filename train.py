import tqdm
from utils import *
from MODEL import *
from torch.optim import Adam
import torch.nn as nn
import os
import h5py


def sinkhorn(pred, lambdas, row, col):
    num_node = pred.shape[0]
    num_class = pred.shape[1]
    p = np.power(pred, lambdas)
    
    u = np.ones(num_node)
    v = np.ones(num_class)

    for index in range(1000):
        u = row * np.power(np.dot(p, v), -1)
        u[np.isinf(u)] = -9e-15
        v = col * np.power(np.dot(u, p), -1)
        v[np.isinf(v)] = -9e-15
    u = row * np.power(np.dot(p, v), -1)
    target = np.dot(np.dot(np.diag(u), p), np.diag(v))
    return target



def train(model, X, y, A, A_norm, Ad, scale_factor):
    """
    train our model
    Args:
        model: Dual Correlation Reduction Network
        X: input feature matrix
        y: input label
        A: input origin adj
        A_norm: normalized adj
        Ad: graph diffusion
    Returns: acc, nmi, ari, f1
    """ 
    print("Training…")
    prediect_y = []
    # calculate embedding similarity and cluster centers
    sim, centers, pseudo_labels = model_init(model, X, y, A_norm)

    # initialize cluster centers
    model.cluster_centers.data = torch.tensor(centers).to(opt.args.device)

    # edge-masked adjacency matrix (Am): remove edges based on feature-similarity
    Am = remove_edge(A, sim, remove_rate=0.1)


    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.epoch)):
        # add gaussian noise to X
        X_tilde1, X_tilde2 = gaussian_noised_feature(X)

        # input & output
        X_hat, Z_hat, A_hat, _, Z_ae_all, Z_gae_all, Z, AZ_all, Z_all, meanbatch, dispbatch, pibatch = model(X_tilde1, Am, X_tilde2, Ad)

        L_ZINB = zinb_loss(X, meanbatch, dispbatch, pibatch, scale_factor)
        L_DICR = x(Z_ae_all, Z_gae_all, AZ_all, Z_all)
        L_REC = reconstruction_loss(X, A_norm, X_hat, Z_hat, A_hat)
        L_SIFM = (L_DICR + L_REC) + opt.args.beta_value* L_ZINB

        #### KL_OPT from scCDCG
        class_assign_model = ClusterAssignment(opt.args.n_clusters, len(Z.T), 1, model.cluster_centers.data).to(opt.args.device)
        temp_class = class_assign_model(Z)


        pseudo_labels_tensor = torch.tensor(pseudo_labels).to(opt.args.device)
        if epoch == 1:
            p_distribution = torch.tensor(sinkhorn(temp_class.cpu().detach().numpy(), opt.args.opt_parameter, torch.ones(np.array(X.shape[0])), [torch.sum(torch.from_numpy(pseudo_labels_tensor.cpu().detach().numpy())==i) for i in range(opt.args.n_clusters)])).float().to(opt.args.device).detach()     
            q_max, q_max_index = torch.max(p_distribution, dim=1)
        elif epoch // 10 == 0:
            p_distribution = torch.tensor(sinkhorn(temp_class.cpu().detach().numpy(), opt.args.opt_parameter, torch.ones(np.array(X.shape[0])), [torch.sum(torch.from_numpy(pseudo_labels_tensor.cpu().detach().numpy()==i)) for i in range(opt.args.n_clusters)])).float().to(opt.args.device).detach()
            q_max, q_max_index = torch.max(p_distribution, dim=1)
        KL_loss_function = nn.KLDivLoss(reduction='sum')                

        
        L_CLU = KL_loss_function(temp_class.to(opt.args.device), p_distribution.to(opt.args.device)) / temp_class.shape[0]

        loss = L_SIFM +  opt.args.lambda_value * L_CLU

        logger.info(f'Epoch {epoch}/{opt.args.epoch}: loss:{loss}; L_DICR:{0.2 * L_DICR}; L_REC:{0.1 * L_REC}; L_KL:{L_CLU}; L_ZINB:{L_ZINB}')

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # clustering & evaluation
        # print(epoch)
        acc, nmi, ari, f1, _, prediect_y = clustering(Z, y)
        logger.info('Epoch {}/{}: ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}'.format(epoch, opt.args.epoch, acc, nmi, ari, f1))
        if acc > opt.args.acc:
            opt.args.acc = acc
            opt.args.nmi = nmi
            opt.args.ari = ari
            opt.args.f1 = f1
            prediect_y = prediect_y

    return opt.args.acc, opt.args.nmi, opt.args.ari, opt.args.f1, Z, prediect_y


