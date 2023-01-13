import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.utils.data import DataLoader
from torchvision import models
import os
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from tqdm import tqdm 
import wandb

from models.discriminator import Discriminator
from vqgan import VQGAN
from data import VQData
from config import get_args


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ContentLoss(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        self.vgg19 = models.vgg19(pretrained = True).features[:35].eval().cuda()
        self.loss = nn.MSELoss()
        for params in self.vgg19.parameters():
            params.requires_grad =False
            
    def forward(self, ip_img: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
        
        ip_features = self.vgg19(ip_img)
        target_features = self.vgg19(target_image)
        vgg_loss = self.loss(ip_features, target_features)
        return vgg_loss



def train_s1(args, model, disc, disc_start, loader, valid_loader, opt, disc_opt, vgg_loss, l1_loss, scaler_g, scaler_d, epoch, total_steps):

    model.train()
    disc.train()
    
    wandb.log({
        'Epoch': epoch
    })

    loss_vgg, loss_gan, loss_l1, total_loss, loss_gen = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    real_loss, fake_loss = AverageMeter(), AverageMeter()
    loop = tqdm(loader, position = 0, leave = True)

    for idx, imgs in enumerate(loop):

        imgs = imgs.cuda(non_blocking = True)

        opt.zero_grad(set_to_none = True)
        disc_opt.zero_grad(set_to_none = True)

        with torch.cuda.amp.autocast():
            
            pred, _, vq_loss = model(imgs)
            if epoch > disc_start:

                disc_real = disc(imgs.contiguous())
                disc_fake = disc(pred.contiguous())
                d_loss_real = torch.mean(fun.relu(1.0 - disc_real))
                d_loss_fake = torch.mean(fun.relu(1.0 + disc_fake))
                gen_loss = -torch.mean(disc_fake)
                disc_factor = model.adopt_weight(1.0, epoch * total_steps + idx, 5000)

            perceptual_loss = vgg_loss(torch.clamp(pred, 0.0, 1.0), torch.clamp(imgs, 0.0, 1.0))
            recon_loss = l1_loss(pred, imgs)
            perceptual_recon_loss = args.vgg_factor * perceptual_loss + args.l1_factor * recon_loss
            perceptual_recon_loss = perceptual_recon_loss.mean() 
            
            
            if epoch > disc_start:
                lambda_ = model.calc_lambda(perceptual_recon_loss, gen_loss)
                quality_loss = perceptual_recon_loss + vq_loss + (disc_factor * lambda_ *  gen_loss)
                gan_loss = 0.5 * (d_loss_real + d_loss_fake)
            
            else:
                quality_loss = perceptual_recon_loss + vq_loss 
        
        scaler_g.scale(quality_loss).backward()
        if epoch > disc_start:
            scaler_d.scale(gan_loss).backward()

        scaler_g.step(opt)
        scaler_g.update()

        if epoch > disc_start:
            scaler_d.step(disc_opt)
            scaler_d.update()     
        
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.5)
        torch.nn.utils.clip_grad.clip_grad_norm_(disc.parameters(), 0.5)

        loss_vgg.update(perceptual_loss.detach(), imgs.size(0))
        loss_l1.update(recon_loss.detach(), imgs.size(0))
        total_loss.update(quality_loss.detach(), imgs.size(0))
        
        if epoch > disc_start:
            loss_gan.update(gan_loss.detach(), imgs.size(0))
            loss_gen.update(gen_loss.detach(), imgs.size(0))
            real_loss.update(d_loss_real.detach(), imgs.size(0))
            fake_loss.update(d_loss_fake.detach(), imgs.size(0))
        

        if idx % 100 == 0:
            if epoch > disc_start:
                wandb.log(
                    {
                        'VGG Loss': loss_vgg.avg,
                        'L1 Loss': loss_l1.avg,
                        'Generator Loss': loss_gen.avg,
                        'Discriminator Real': real_loss.avg,
                        'Discriminator Fake': fake_loss.avg,
                        'GAN Loss': loss_gan.avg,
                        'Quality Loss': total_loss.avg
                    }
                )
            else:
                wandb.log(
                    {
                        'VGG Loss': loss_vgg.avg,
                        'L1 Loss': loss_l1.avg,
                        'Quality Loss': total_loss.avg
                    }
                )

            wb_img = wandb.Image(imgs[0].detach().cpu().numpy().transpose(1,2,0))
            wb_pred = wandb.Image(pred[0].detach().cpu().numpy().transpose(1,2,0))

            wandb.log({
                'Input': wb_img,
                'Reconstruction': wb_pred
            })

        
def main():

    args = get_args()
    #disc_start represents the epoch at which we start training the discriminator.
    disc_start = 5               
    train_data = VQData(args.train_data_path)
    train_loader = DataLoader(train_data, args.batchsize, shuffle = True, pin_memory = True, num_workers = 0)
    valid_loader = ''
    total_steps = train_data.__len__() // args.batchsize

    model = VQGAN(args).cuda()
    disc = Discriminator().cuda()

    #if args.model_path is not None or args.model_path != '':
    #    model.load_state_dict(torch.load(args.model_path))


    # torchmetric's LPIPS uses pretrained VGG16 model 
    vgg_loss = LearnedPerceptualImagePatchSimilarity('vgg').cuda()
    # vgg_loss = ContentLoss()
    # psnr = PeakSignalNoiseRatio()
    l1_loss = nn.L1Loss()

    opt = torch.optim.Adam(model.parameters(), lr = args.gen_lr, betas = (0.5, 0.999))
    disc_opt = torch.optim.Adam(disc.parameters(), lr = args.disc_lr, betas = (0.5, 0.999) )
    scaler_g = torch.cuda.amp.grad_scaler.GradScaler()
    scaler_d = torch.cuda.amp.grad_scaler.GradScaler()

    for epoch in range(1, args.num_epochs + 1):

        train_s1(args, model, disc, disc_start, train_loader, valid_loader, opt, disc_opt, vgg_loss, l1_loss, scaler_g, scaler_d, epoch, total_steps)
        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'vqgan.pth'))
        if epoch > disc_start:
            torch.save(disc.state_dict(), os.path.join(args.ckpt_dir, 'discriminator.pth'))

if __name__ == '__main__':

    wandb.init(project = 'VQGAN')
    main()
