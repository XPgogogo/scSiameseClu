import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F


class AE_encoder(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, n_input, n_z):
        super(AE_encoder, self).__init__()
        self.enc_1 = Linear(n_input, ae_n_enc_1)
        self.enc_2 = Linear(ae_n_enc_1, ae_n_enc_2)
        self.enc_3 = Linear(ae_n_enc_2, ae_n_enc_3)
        self.z_layer = Linear(ae_n_enc_3, n_z)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        z = self.act(self.enc_1(x))
        z = self.act(self.enc_2(z))
        z = self.act(self.enc_3(z))
        z_ae = self.z_layer(z)
        return z_ae


class AE_decoder(nn.Module):

    def __init__(self, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE_decoder, self).__init__()

        self.dec_1 = Linear(n_z, ae_n_dec_1)
        self.dec_2 = Linear(ae_n_dec_1, ae_n_dec_2)
        self.dec_3 = Linear(ae_n_dec_2, ae_n_dec_3)
        self.x_bar_layer = Linear(ae_n_dec_3, n_input)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # ZINB编码器的最后一层
        # self._dec_mean = nn.Sequential(nn.Linear(ae_n_dec_3, n_input), MeanAct())
        # self._dec_disp = nn.Sequential(nn.Linear(ae_n_dec_3, n_input), DispAct())
        # self._dec_pi = nn.Sequential(nn.Linear(ae_n_dec_3, n_input), nn.Sigmoid())

    # def forward(self, z_ae):
    #     z = self.act(self.dec_1(z_ae))
    #     z = self.act(self.dec_2(z))
    #     z = self.act(self.dec_3(z))


    #     # ZINB AE decoding
    #     _mean = self._dec_mean(z)
    #     _disp = self._dec_disp(z)
    #     _pi = self._dec_pi(z)

    #     x_hat = self.x_bar_layer(z)
    #     return x_hat, _mean, _disp, _pi
    def forward(self, z_ae):
        z = self.act(self.dec_1(z_ae))
        z = self.act(self.dec_2(z))
        z = self.act(self.dec_3(z))

        x_hat = self.x_bar_layer(z)
        return x_hat

class AE(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3, ae_n_dec_1, ae_n_dec_2, ae_n_dec_3, n_input, n_z):
        super(AE, self).__init__()

        self.encoder = AE_encoder(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            n_input=n_input,
            n_z=n_z)

        self.decoder = AE_decoder(
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)

    # def forward(self, x):
    #     z_ae = self.encoder(x)
    #     x_hat, _mean, _disp, _pi = self.decoder(z_ae)
    #     return x_hat, z_ae, _mean, _disp, _pi 
    def forward(self, x):
        z_ae = self.encoder(x)
        x_hat = self.decoder(z_ae)
        return x_hat, z_ae