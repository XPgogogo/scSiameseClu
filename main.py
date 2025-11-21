from train import *
from MODEL import scSiameseClu

from time import time
import warnings
warnings.filterwarnings('ignore')
import h5py
import os
import numpy as np
from torch import nn



if __name__ == '__main__':

    for i in range(1):
        # setup
        setup()
        torch.cuda.set_device(opt.args.gpu)

        # data pre-precessing: X, y, A, A_norm, Ad
        X, y, A, sf = load_graph_data(opt.args.name, show_details=False)
        A_norm = normalize_adj(A, self_loop=True, symmetry=False)
        Ad = diffusion_adj(A, mode="ppr", transport_rate=opt.args.alpha_value)
        
        # to torch tensor
        X = numpy_to_torch(X).to(opt.args.device)
        A_norm = numpy_to_torch(A_norm, sparse=True).to(opt.args.device)
        Ad = numpy_to_torch(Ad).to(opt.args.device)


        model = scSiameseClu(n_node=X.shape[0]).to(opt.args.device)

        # deep graph clustering
        start_time = time()
        acc, nmi, ari, f1, Z, prediect_y = train(model, X, y, A, A_norm, Ad, sf)
        use_time = time() - start_time

        print("MAX_ACC: {:.4f},".format(acc), "NMI: {:.4f},".format(nmi), "ARI: {:.4f},".format(ari), "F1: {:.4f}".format(f1), "Use_time:{}".format(use_time))
        logger.info('maxACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}, F1: {:.4f}, Use_time: {:10f}'.format(acc, nmi, ari, f1, use_time))
