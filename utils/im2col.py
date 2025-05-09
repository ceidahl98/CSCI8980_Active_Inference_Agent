from math import sqrt
import torch
import numpy as np
def collect_latents(i,j,x,x_):

    x[:, :, i, j] = x_.reshape(x.shape[0], x.shape[1], -1, 12)
    return x

def dispatch_latents(i,j,x):
    i = np.array(i)
    j = np.array(j)
    x_ = x[:, :, i, j].reshape(x.shape[0], x.shape[1], i.shape[0], i.shape[1])

    return x, x_

def dispatch(i, j, x):
    # i, j are indices of elements being processed
    # dim(i) = (num_RG_blocks, K*K)
    # dim(x_) = (B, C, num_RG_blocks, K*K)
    if len(x.shape) == 4:
        x_ = x[:, :, i, j].reshape(x.shape[0], x.shape[1], i.shape[0], i.shape[1])
    else:
        x_ = x[:, :, i, j,:].reshape(x.shape[0], x.shape[1], i.shape[0], i.shape[1],-1)

    return x, x_

def dispatch_3d(t_idx, i_idx, j_idx, x):
    """
    Returns
    -------
    x      : untouched reference (so your loop signature stays the same)
    x_patch: (B, C, M, P)  advanced-indexed tensor
    """
    device = x.device
    t = torch.as_tensor(np.array(t_idx), device=device, dtype=torch.long)
    i = torch.as_tensor(np.array(i_idx), device=device, dtype=torch.long)
    j = torch.as_tensor(np.array(j_idx), device=device, dtype=torch.long)

    # Advanced indexing: result has shape  (B, C, M, P)
    x_patch = x[:, :, t, i, j]          # broadcast over (B,C)

    return x, x_patch





def collect(i, j, x, x_):

    if len(x.shape) == 4:
        x = x.clone()
        x[:, :, i, j] = x_.reshape(x.shape[0], x.shape[1], i.shape[0], i.shape[1])
    else:
        x = x.clone()
        x[:, :, i, j,:] = x_.reshape(x.shape[0], x.shape[1], i.shape[0], i.shape[1],-1)
    return x

def collect_3d(t_idx, i_idx, j_idx, x, x_patch):
    device = x.device
    t = torch.as_tensor(t_idx, device=device, dtype=torch.long)
    i = torch.as_tensor(i_idx, device=device, dtype=torch.long)
    j = torch.as_tensor(j_idx, device=device, dtype=torch.long)

    out = x.clone()                     # keep autograd graph intact

    out[:, :, t, i, j] = x_patch        # in-place scatter
    return out




def stackRGblock(x):
    # x should be dispatched
    # dim(x) = (B, C, num_RG_blocks, K*K)
    # -> (B, num_RG_blocks, C, K*K)
    # -> (B*num_RG_blocks, C, K, K)

    _, C, _, KK = x.shape
    K = int(sqrt(KK))
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(-1, C, K, K)


    return x

def stackRGblock_3d(x,t,patch_size):
    _,C,_,P = x.shape

    K = patch_size
    x = x.permute(0,2,1,3)
    x = x.reshape(-1,int(C*t),K,K)
    return x

def unstackRGblock(x, batch_size):
    # dim(x) = (B*num_RG_blocks, C, K, K)
    # -> (B, num_RG_blocks, C, K*K)
    # -> (B, C, num_RG_blocks, K*K)
    if len(x.shape) == 4:
        _, C, KH, KW = x.shape
        x = x.reshape(batch_size, -1, C, KH * KW)
        x = x.permute(0, 2, 1, 3)
    else:
        _, C, KH, KW,clusters = x.shape
        x = x.reshape(batch_size, -1, C, KH * KW,clusters)
        x = x.permute(0, 2, 1, 3,4)
    return x


def unstackRGBlock_3d(x, batch_size):
    B_times_M, C_tot, K, _ = x.shape
    M         = B_times_M // batch_size
    t_kernel  = (x.numel() // B_times_M) // (K*K)
    C_orig    = C_tot // t_kernel
    x = x.view(batch_size, int(M), int(C_orig), int(t_kernel * K * K)).permute(0, 2, 1, 3)
    return x                        # (B, C_orig, M, P)

