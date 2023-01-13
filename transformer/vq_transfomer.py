import torch
import torch.nn as nn
import torch.nn.functional as fun

from mingpt import MinGPT
from ..vqgan import VQGAN


class Transformer(nn.Module):

    def __init__(self, args):

        super().__init__()
        self.args = args
        self.vqgan = VQGAN(args)
        self.vqgan.load_state_dict(torch.load(args.model_path))
        self.vqgan.eval()
        self.gpt = MinGPT(args)

    @torch.no_grad()
    def encode(self, x):

        z, idxs, _ = self.vqgan.encode(x)
        idxs = idxs.view(z.shape[0], -1)
        return z, idxs
    
    @torch.no_grad()
    def z_to_img(self, idx, p1 = 16, p2 = 16):

        idx_to_vec = self.vqgan.vq.embs(idx).reshape(idx.shape[0], p1, p2, 256)
        idx_to_vec = idx_to_vec.permute(0,3,1,2)
        img = self.vqgan.decode(idx_to_vec)
        return img

    def forward(self, x):

        _, idx = self.encode(x)
        sos_tokens = torch.ones(x.shape[0], 1) * self.args.sos_token
        sos_tokens = sos_tokens.long().cuda()

        mask = torch.bernoulli(self.args.mask_prob * torch.ones(idx.shape, device = idx.device))
        mask = mask.round().to(torch.int64)

        random_idx = torch.randint_like(idx, self.args.d)
        new_idxs = mask * idx + (1 - mask) * random_idx
        new_idxs = torch.cat([sos_tokens, new_idxs], dim = 1)
        target = idx

        res, _ = self.gpt(new_idxs[:, :-1])
        return res, target

    def top_k_logits(self, logits, k):

        res, idx = torch.topk(logits, k)
        out = logits.clone()
        out[out < res[..., [-1]]] = float('-inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temp = 1.0, topk = 100):

        self.gpt.eval()
        x = torch.cat([c, x], dim = 1)

        for i in range(steps):
            logits, _ = self.gpt(x)
            logits = logits[:, -1, :] / temp

            if topk is not None:
                logits = self.top_k_logits(logits, topk)
            prob = fun.softmax(logits, dim = -1)
            ix = torch.multinomial(prob, num_samples = 1)
            x = torch.cat([x, ix], dim = 1)
        
        x = x[:, c.shape[1]:]
        self.gpt.train()
        return x

