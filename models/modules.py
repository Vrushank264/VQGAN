import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as fun
from torchsummary import summary


'''
Conatins:

VQVAE Encoder
Quantiazation
Transformer as a sequence predictor
CNN Decoder
CNN Patchwise Discriminator

'''

def init_weights(m):
    
    cls_name = m.__class__.__name__
    if cls_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif cls_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups = 32, num_channels = in_c),
            Swish(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding = 1),
            nn.GroupNorm(num_groups = 32, num_channels = out_c),
            Swish(),
            nn.Conv2d(out_c, out_c, kernel_size = 3, padding = 1)
        )

        self.skip_connection = nn.Conv2d(in_c, out_c, kernel_size = 1, padding = 0)


    def forward(self, x):

        if self.in_c != self.out_c:
            return self.skip_connection(x) + self.block(x)

        else:
            return x + self.block(x)


class UpSample(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
    
    def forward(self, x):

        x = fun.interpolate(x, scale_factor = 2.0)
        return self.conv(x)


class DownSample(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, stride = 2, padding = 0)

    def forward(self, x):

        val = (0,1,0,1)
        x = fun.pad(x, val, mode = 'constant', value = 0)
        return self.conv(x)


class SelfAttention(nn.Module):

    def __init__(self, dim, heads):

        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dims = self.dim // self.heads
        self.scale = self.head_dims ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias = False)
        self.attn_dropout = nn.Dropout(0.2)
        self.projection = nn.Linear(dim, dim)
    

    def forward(self, x):

        num_samples, num_tokens, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(num_samples, num_tokens, 3, self.heads, self.head_dims)
        qkv = qkv.permute(2,0,3,1,4)
        q, k ,v = qkv[0], qkv[1], qkv[2]
        key_t = k.transpose(-2, -1)

        dot_prod = (q @ key_t) * self.scale
        attn = dot_prod.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        self_attn = attn @ v
        self_attn = self_attn.transpose(1, 2).flatten(2)
        out = self.projection(self_attn)

        return out


class FasterAttention(nn.Module):

    '''
    Implement Faster Attention as well and compare it with Normal Attention.
    '''
    pass


class NonLocalBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()
        
        self.in_c = channels
        self.GroupNorm = nn.GroupNorm(32, self.in_c)
        self.QKV = nn.Conv2d(self.in_c, self.in_c * 3, kernel_size = 1, padding = 0)
        self.projection = nn.Conv2d(self.in_c, self.in_c, kernel_size = 1, padding = 0)

    def forward(self, x):

        out = self.GroupNorm(x)
        b, c, h, w = out.shape
        scale = c ** -0.5 

        qkv = self.QKV(out)
        qkv = qkv.reshape(b, c, 3, h*w)
        qkv = qkv.permute(2,0,1,3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k.permute(0,2,1)
        attn = torch.bmm(q, k) * scale
        attn = fun.softmax(attn, dim = -1)
        attn = attn.permute(0,2,1)

        self_attn = torch.bmm(attn, v)
        out = self_attn.reshape(b, c, h, w)

        return x + out



class Encoder(nn.Module):

    def __init__(self, args, channels = [64, 128, 256, 256, 512]):
        
        super().__init__()
        self.num_res_blks = args.num_res_blks 
        #self.num_res_blks = num_res_blks
        self.channels = channels
        self.attn_res = [16]
        self.first_layer = nn.Conv2d(3, self.channels[0], kernel_size = 1, padding = 1)
        self.penultimate_blk = nn.Sequential(ResidualBlock(channels[-1], channels[-1]),
                                            NonLocalBlock(channels[-1]),
                                            ResidualBlock(channels[-1], channels[-1]),
                                            nn.GroupNorm(32, channels[-1]),
                                            Swish()
                                            )
        
        self.final_layer = nn.Conv2d(channels[-1], args.latent_dim, kernel_size = 3, padding = 1)
        #self.final_layer = nn.Conv2d(channels[-1], 256, kernel_size = 3, padding = 1)
        self.encoder = self.build_encoder()


    def build_encoder(self, resolution = 256):

        layers = []
        layers.append(self.first_layer)

        for i in range(len(self.channels) - 1):
            in_c = self.channels[i]
            out_c = self.channels[i + 1]

            for j in range(self.num_res_blks):
                
                layers.append(ResidualBlock(in_c, out_c))
                in_c = out_c
                if resolution in self.attn_res:
                    layers.append(NonLocalBlock(in_c))
                
            if i != len(self.channels) - 2:
                layers.append(DownSample(self.channels[i+1]))
                resolution = resolution // 2
        
        layers.append(self.penultimate_blk)
        layers.append(self.final_layer)

        enc = nn.Sequential(*layers)
        return enc
    
    
    def forward(self, x):

        return self.encoder(x)


class Decoder(nn.Module):

    def __init__(self, num_res_blks = 3, channels = [512, 256, 256, 128]):
        
        super().__init__()
        #self.args = args
        self.channels = channels
        #self.num_res_blks = args.num_res_blks_dec
        self.num_res_blks = num_res_blks
        self.attn_res = [16]
        self.first_layer = nn.Sequential(nn.Conv2d(256, channels[0], kernel_size = 3, padding = 1),
                                        ResidualBlock(channels[0], channels[0]),
                                        NonLocalBlock(channels[0]),
                                        ResidualBlock(channels[0], channels[0])
                                        )

        self.final_layer = nn.Conv2d(channels[-1], 3, kernel_size = 3, padding = 1)
        self.decoder = self.build_decoder()


    def build_decoder(self, resolution = 16):

        layers = []
        layers.append(self.first_layer)
        in_c = self.channels[0]

        for i in range(len(self.channels)):
            
            out_c = self.channels[i]

            for j in range(self.num_res_blks):
                layers.append(ResidualBlock(in_c, out_c))
                in_c = out_c
                if resolution in self.attn_res:
                    layers.append(NonLocalBlock(in_c))
            
            if i != 0:
                layers.append(UpSample(in_c))
                resolution *= 2

        layers.append(nn.GroupNorm(32, in_c))
        layers.append(Swish())

        layers.append(self.final_layer)
        model = nn.Sequential(*layers)
        return model
        

    def forward(self, x):

        return self.decoder(x)

# args = get_args()
# model = Encoder(args)
# #x = torch.randn((1,3,256,256))
# #res = model(x)

# print(summary(model, (3,256,256), device = 'cpu'))
#print(res.shape)