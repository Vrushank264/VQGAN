import torch
import torch.nn as nn
import torch.nn.functional as fun
from torch.autograd import Function


class VectorQuantization(nn.Module):

    def __init__(self, args):

        super().__init__()
        self.k = args.k
        self.d = args.d
        self.beta = args.beta

        self.embs = nn.Embedding(self.k, self.d)
        self.embs.weight.data.uniform_(-1.0 / self.k, 1.0 / self.k)

    def forward(self, z):

        z = z.permute(0,2,3,1).contiguous()
        z_flat = z.view(-1, self.d)
        
        dis = torch.cdist(z_flat, self.embs.weight) ** 2
        min_idxs = torch.argmin(dis, dim = 1)
        z_q = self.embs(min_idxs).view(z.shape)
        vq_loss = fun.mse_loss(z, z_q.detach())
        commit_loss = fun.mse_loss(z_q, z.detach())
        loss = vq_loss + self.beta * commit_loss

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0,3,1,2)

        return z_q, min_idxs, loss
    

# class VectorQuantization(Function):

#     @staticmethod
#     def forward(ctx, inputs, codebook):

#         with torch.no_grad():

#             ip_size = inputs.size()
#             emb_size = codebook.size(1)
#             flat_ip = inputs.view(-1, emb_size)

#             codebook_sq = torch.sum(codebook ** 2, dim = 1)
#             ip_sq = torch.sum(flat_ip ** 2, dim = 1, keepdim = True)
#             l2_dis = torch.addmm(input = codebook_sq + ip_sq, mat1 = flat_ip,
#                                 mat2 = codebook.t(), alpha = 2.0, beta = 1.0)
#             print(ip_size, flat_ip.shape, emb_size, ip_size)
#             print('*********************************************************')
#             print(l2_dis.shape)
#             print('*********************************************************')
#             idx_flatten = torch.argmin(l2_dis, dim = 1)
#             print(idx_flatten.shape)
#             print('*********************************************************')
#             idx = idx_flatten.view(*ip_size[-3:-1])
#             ctx.mark_non_differentiable(idx)

#             return idx, idx_flatten

#     @staticmethod    
#     def backward(ctx, grad_outpus):

#         raise RuntimeError('Trying to call backward on graph containing `Vector Quantization`' 
#                             'which is non-differentiable, Call VQStraightThrough (VQST) instead.')



# class VQStraightThrough(Function):

#     @staticmethod
#     def forward(ctx, inputs, codebook):

#         idx, idx_flatten = VQ(inputs, codebook)
#         flat_idx = idx.view(-1)
#         ctx.save_for_backward(flat_idx, codebook)
#         ctx.mark_non_differentiable(flat_idx)
#         codes_flatten = torch.index_select(input = codebook, dim = 0, index = flat_idx)
#         codes = codes_flatten.view_as(inputs)

#         return (codes, flat_idx, idx_flatten)

#     @staticmethod
#     def backward(ctx, grad_outputs, grad_indices):

#         grad_inputs, grad_codebook = None, None

#         if ctx.needs_input_grad[0]:

#             grad_inputs = grad_outputs.clone()

#         if ctx.needs_input_grad[1]:

#             idx, codebook = ctx.saved_tensors
#             emb_size = codebook.size(1)
#             flat_grad_op = (grad_outputs.contiguous().view(-1, emb_size))
#             grad_codebook = torch.zeros_like(codebook)
#             grad_codebook.index_add_(0, idx, flat_grad_op)

#         return (grad_inputs, grad_outputs)

# VQ = VectorQuantization.apply
# VQ_ST = VQStraightThrough.apply


# class VQEmbedding(nn.Module):

#     def __init__(self, args):

#         super().__init__()
#         self.K = args.k
#         self.D = args.d
#         self.vq_emb = nn.Embedding(self.K, self.D)
#         self.vq_emb.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

#     def forward(self, z):

#         z = z.permute(0,2,3,1).contiguous()
#         latents, idx_flatten = VQ(z, self.vq_emb.weight)

#         return latents, idx_flatten

#     def straight_through_forward(self, z):

#         z = z.permute(0,2,3,1).contiguous()
#         z_q, idx, idx1 = VQ_ST(z, self.vq_emb.weight.detach())
#         z_q = z_q.permute(0,3,1,2).contiguous()
#         flat_zq_tilde = torch.index_select(self.vq_emb.weight, dim = 0, index = idx)
#         zq_tilde = flat_zq_tilde.view_as(z)
#         zq_tilde = zq_tilde.permute(0,3,1,2).contiguous()

#         return z_q, zq_tilde, idx1