import torch
import torch.nn as nn
import torch.nn.functional as fun
from torchsummary import summary

from models.modules import Encoder, Decoder
from models.discriminator import Discriminator
from models.quantization import VectorQuantization
from config import get_args


class VQGAN(nn.Module):

    def __init__(self, args):
        
        super().__init__()
        self.args = args
        self.encoder = Encoder(args)
        self.decoder = Decoder()
        self.vq = VectorQuantization(args)
        #self.vq = VQEmbedding(args)
        self.pre_quant = nn.Conv2d(args.latent_dim, args.latent_dim, kernel_size = 1)
        self.post_quant = nn.Conv2d(args.latent_dim, args.latent_dim, kernel_size = 1)

    
    def forward(self, x):

        z = self.encoder(x)
        res = self.pre_quant(z)
        res, codebook_idx, vq_loss = self.vq(res)
        #zq_st, z_q_x, codebook_idx = self.vq.straight_through_forward(res)
        #vq_loss = fun.mse_loss(z, z_q_x.detach()) + fun.mse_loss()
        #loss = torch.mean((z_q_x.detach() - z) ** 2) + self.args.beta * torch.mean((z_q_x - z.detach()) ** 2)
        res = self.post_quant(res)
        dec_out = self.decoder(res)

        return dec_out, codebook_idx, vq_loss

    def encode(self, x):

        enc_out = self.encoder(x)
        res = self.pre_quant(enc_out)
        res, codebook_idx, vq_loss  = self.vq(res)
        return res, codebook_idx, vq_loss

    def decode(self, z):

        z = self.post_quant(z)
        dec_out = self.decoder(z)
        return dec_out

    def calc_lambda(self, vgg_loss, g_loss):

        last_layer = self.decoder.decoder[-1]
        last_layer_weight = last_layer.weight
        vgg_loss_grads = torch.autograd.grad(vgg_loss, last_layer_weight, retain_graph = True)[0]
        gan_loss_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph = True)[0]
        
        lambda_ = torch.norm(vgg_loss_grads) / (torch.norm(gan_loss_grads) + 1e-6)
        lambda_ = torch.clamp(lambda_, 0, 1e4).detach()

        return lambda_

    @staticmethod
    def adopt_weight(factor, i, threshold, value=0.):
        if i < threshold:
            factor = value
        return factor

# args = get_args()
# model = VQGAN(args)
# x = torch.randn((1,3,64,64))
# res = model(x)[0]
# print(res.shape)
#print(summary(model, (3,256,256), device = 'cpu'))