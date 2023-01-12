import os
import argparse


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type = str, default = '/content/train')
    parser.add_argument('--valid_data_path', type = str, default = '')
    parser.add_argument('--batchsize', type = int, default = 64)
    parser.add_argument('--num_epochs', type = int, default = 50)
    parser.add_argument('--latent_dim', type = int, default = 256)
    parser.add_argument('--k', type = int, default = 256)
    parser.add_argument('--d', type = int, default = 512)
    parser.add_argument('--beta', type = float, default = 1.0)
    parser.add_argument('--num_res_blks', type = int, default = 3)
    parser.add_argument('--gen_lr', type = float, default = 3e-4)
    parser.add_argument('--disc_lr', type = float, default = 2e-5)
    parser.add_argument('--vgg_factor', type = float, default = 1.0)
    parser.add_argument('--l1_factor', type = float, default = 2.0)
    parser.add_argument('--ckpt_dir', type = str, default = '/content/drive/VQGAN')
    parser.add_argument('--model_path', type = str, default = '')
    parser.add_argument('--disc_path', type = str, default = '')
    
    args = parser.parse_args()
    return args