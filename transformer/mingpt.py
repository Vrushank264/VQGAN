import torch
import torch.nn as nn
import torch.nn.functional as fun
import math


class SelfAttention(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.head_dim = args.emb_dim // args.attn_heads
        self.qkv = nn.Linear(args.emb_dim, args.emb_dim * 3)
        self.attn_drop = nn.Dropout(args.attn_drop_p)
        self.project = nn.Linear(args.emb_dim, args.emb_dim)
        self.res_drop = nn.Dropout(args.res_drop_p)

        mask = torch.tril(torch.ones(args.attn_blk_size, args.attn_blk_size))
        self.register_buffer('mask', mask.view(1, 1, args.attn_blk_size, args.attn_blk_size))


    def forward(self, x, layer_past = None):

        num_samples, num_tokens, dims = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(num_samples, num_tokens, 3, self.args.attn_heads, self.head_dims)
        qkv = qkv.permute(2,0,3,1,4)
        q, k ,v = qkv[0], qkv[1], qkv[2]

        present = torch.cat([k, v])
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim = -2)
            v = torch.cat([past_v, v], dim = -2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            attn = attn.masked_fill(self.mask[:, :, :num_tokens, :num_tokens] == 0, float('-inf'))

        attn = fun.softmax(attn, dim = -1)
        attn = self.attn_drop(attn)
        self_attn = attn @ v
        self_attn = self_attn.transpose(1, 2).contiguous().view(num_samples, num_tokens, dims)

        self_attn = self.res_drop(self_attn)
        return self_attn, present


class GPTBlock(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.ln = nn.LayerNorm(args.emb_dim)
        self.sa = SelfAttention(args)
        self.mlpblk = nn.Sequential(nn.Linear(args.emb_dim, 4 * args.emb_dim),
                                    nn.GELU(),
                                    nn.Linear(4 * args.emb_dim, args.emb_dim),
                                    nn.Dropout(args.res_drop_p)
                                    )

        
    def forward(self, x, layer_past = None, return_present = False):

        attn, present = self.sa(self.ln(x), layer_past = layer_past)
        x = x + attn
        x = x + self.mlpblk(self.ln(x))

        if layer_past is not None or return_present:
            return x, present

        return x


class MinGPT(nn.Module):

    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.token_emb = nn.Embedding(args.latent_dim, args.emb_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.attn_blk_size, self.emb_dim))
        self.dropout = nn.Dropout(args.emb_drop_p)
        
        self.blocks = nn.Sequential(*[GPTBlock(args) for _ in range(args.num_heads)])

        self.ln = nn.LayerNorm(args.emb_dim)
        self.head = nn.Linear(args.emb_dim, args.latent_dim, bias = False)
        self.apply(self._init_weights)

    
    def _init_weights(self, m):

        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean = 0.0, std = 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.weight.data.zero_()
        
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            m.weight.bias.zero_()

    
    def forward(self, x, embs = None):

        token_embs = self.token_emb(x)
        if embs is not None:
            token_embs = torch.cat([token_embs, embs], dim = 1)
        
        t = token_embs.shape[1]
        pos_embs = self.pos_emb[: :t, :]
        res = self.drop(token_embs + pos_embs)
        res = self.blocks(res)
        res = self.head(self.ln(res))

        return res
