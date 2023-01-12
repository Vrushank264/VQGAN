import torch
import torch.nn as nn
from torchsummary import summary


class Block(nn.Module):

    def __init__(self, in_c, out_c, stride = 2):
        super().__init__()

        self.blk = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size = 4, stride = stride, padding = 1),
                                nn.BatchNorm2d(out_c),
                                nn.LeakyReLU(0.2)
                                )
        
    def forward(self, x):

        return self.blk(x)


class Discriminator(nn.Module):

    def __init__(self, channels = [64, 128, 256, 512]):
        super().__init__()  

        self.channels = channels
        self.first_layer = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size = 4, stride = 2, padding = 1),
                                        nn.LeakyReLU(0.2))
        self.last_layer = nn.Conv2d(channels[-1], 1, kernel_size = 4, padding = 1)
        self.discriminator = self.build_disc()


    def build_disc(self):

        layers = []
        layers.append(self.first_layer)
        in_c = self.channels[0]
        for i in self.channels[1:]:
            layers.append(Block(in_c, i, stride = 1 if i == self.channels[-1] else 2))
            in_c = i
        layers.append(self.last_layer)
        model = nn.Sequential(*layers)
        return model       


    def forward(self, x):

        return self.discriminator(x)

 

# model = Discriminator()
# x = torch.randn((1, 3, 256, 256))
# res = model(x)
# print(summary(model, (3, 256, 256), device = 'cpu'))
# print(res)


