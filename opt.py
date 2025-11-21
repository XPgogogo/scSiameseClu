import argparse

parser = argparse.ArgumentParser(description='scSiameseClu', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu', default=3, type=int)
# setting

parser.add_argument('--name', type=str, default='Meuro_human_Pancreas_cell')
parser.add_argument('--n_clusters', type=int, default=9)

parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--alpha_value', type=float, default=0.5)
parser.add_argument('--sigma_value', type=float, default=5)  # ZINB structure parameter from scDSC
parser.add_argument('--mu_value', type=float, default=50)
parser.add_argument('--delta_value', type=float, default=1e3)
parser.add_argument('--opt_parameter', type=int, default=5) # lambda
parser.add_argument('--n_z', type=int, default=20)
parser.add_argument('--show_training_details', type=bool, default=False)
parser.add_argument('--n_input', type=int, default=100)

# AE structure parameter from DFCN
parser.add_argument('--ae_n_enc_1', type=int, default=128)
parser.add_argument('--ae_n_enc_2', type=int, default=256)
parser.add_argument('--ae_n_enc_3', type=int, default=512)
parser.add_argument('--ae_n_dec_1', type=int, default=512)
parser.add_argument('--ae_n_dec_2', type=int, default=256)
parser.add_argument('--ae_n_dec_3', type=int, default=128)

# IGAE structure parameter from DFCN
parser.add_argument('--gae_n_enc_1', type=int, default=128)
parser.add_argument('--gae_n_enc_2', type=int, default=256)
parser.add_argument('--gae_n_enc_3', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=256)
parser.add_argument('--gae_n_dec_3', type=int, default=128)

# clustering performance: acc, nmi, ari, f1
parser.add_argument('--acc', type=float, default=0)
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--f1', type=float, default=0)

args = parser.parse_args()

