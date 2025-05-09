from math import log

import numpy as np

from .hierarchy import HierarchyBijector


def get_indices(shape, height, width, stride, dialation, offset):
    H, W = shape
    out_height = (H - dialation * (height - 1) - 1) // stride + 1
    out_width = (W - dialation * (width - 1) - 1) // stride + 1

    i0 = np.repeat(np.arange(height) * dialation, width)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(width) * dialation, height)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    return (i.transpose(1, 0) + offset) % H, (j.transpose(1, 0) + offset) % W

def get_indices_3d(shape, kernel, stride, dilation, offset):
    T, H, W = shape
    k_t, k_h, k_w = kernel
    s_t, s_h, s_w = stride
    d_t, d_h, d_w = dilation
    o_t, o_h, o_w = offset

    out_T = (T - d_t * (k_t - 1) - 1) // s_t + 1
    out_H = (H - d_h * (k_h - 1) - 1) // s_h + 1
    out_W = (W - d_w * (k_w - 1) - 1) // s_w + 1

    # local offsets (flattened over the patch)
    t0 = np.repeat(np.arange(k_t) * d_t, k_h * k_w)
    y0 = np.tile(np.repeat(np.arange(k_h) * d_h, k_w), k_t)
    x0 = np.tile(np.arange(k_w) * d_w, k_t * k_h)

    # global patch-start positions (flattened over the volume)
    t1 = s_t * np.repeat(np.arange(out_T), out_H * out_W)
    y1 = s_h * np.tile(np.repeat(np.arange(out_H), out_W), out_T)
    x1 = s_w * np.tile(np.arange(out_W), out_T * out_H)

    # combine local + global + offset
    t = t0.reshape(-1, 1) + t1.reshape(1, -1) + o_t
    y = y0.reshape(-1, 1) + y1.reshape(1, -1) + o_h
    x = x0.reshape(-1, 1) + x1.reshape(1, -1) + o_w

    # wrap and transpose to (num_patches, patch_size)
    t_idx = (t % T).transpose(1, 0)
    y_idx = (y % H).transpose(1, 0)
    x_idx = (x % W).transpose(1, 0)

    return t_idx, y_idx, x_idx


def mera_indices(L, kernel_size):
    index_list = []
    depth = int(log(L / kernel_size, 2) + 1)
    for i in range(depth):
        index_list.append(
            get_indices([L, L], kernel_size, kernel_size, kernel_size * 2**i,
                        2**i, 0))
        index_list.append(
            get_indices([L, L], kernel_size, kernel_size, kernel_size * 2**i,
                        2**i, kernel_size * 2**i // 2))
    indexI = [item[0] for item in index_list]
    indexJ = [item[1] for item in index_list]


    print(len(indexI),"INDEX")

    latentI = [list(layer) for layer in indexI]  # Convert to nested lists
    latentJ = [list(layer) for layer in indexJ]
    for d in range(len(latentI)//2-1):
        layer = 2 * (d + 1) - 1
        mod = 2 ** (d + 1)

        for i in range(len(latentI[layer])):
            filtered_pairs = [
                (val, valJ) for val, valJ in zip(latentI[layer][i], latentJ[layer][i])
                if not (val % mod == 0 and valJ % mod == 0)
            ]

            # Unzip the filtered pairs back into latentI and latentJ
            latentI[layer][i], latentJ[layer][i] = zip(*filtered_pairs) if filtered_pairs else ([], [])

    return indexI, indexJ, latentI,latentJ


def mera_indices_3d(L, kernel_size=2, seq_length=4):
    """
    Compute multi-level, time-only RG indices in the same style as mera_indices.

    Returns:
      indexI3d, indexJ3d, indexK3d: lists of time, y, x arrays
      latentI3d, latentJ3d, latentK3d: their complements (the latent slots)
    """
    index_list = []
    # determine number of levels
    depth3d = int(log(seq_length / kernel_size, 2) + 1)

    # collect coarse indices for each level (two phases: no offset and half-offset)
    for d in range(depth3d):
        stride_t = kernel_size * (2 ** d)
        stride = (stride_t, 4, 4)
        dil = (2 ** d, 1, 1)
        for off_t in (0, stride_t // 2):
            t_idx, y_idx, x_idx = get_indices_3d(
                (seq_length, L, L),
                (kernel_size, 4,4),
                stride,
                dil,
                (off_t, 0, 0)
            )
            index_list.append((t_idx, y_idx, x_idx))

    # split into separate lists
    indexI3d = [item[0] for item in index_list]
    indexJ3d = [item[1] for item in index_list]
    indexK3d = [item[2] for item in index_list]

    latentI3d = [list(layer) for layer in indexI3d]  # deep copy
    latentJ3d = [list(layer) for layer in indexJ3d]
    latentK3d = [list(layer) for layer in indexK3d]

    for d in range(len(latentI3d) // 2 - 1):  # every “offset” phase
        layer = 2 * (d + 1) - 1  # (same formula as 2-D)
        mod = 2 ** (d + 1)  # stride in the time axis

        for p in range(len(latentI3d[layer])):  # each patch in this layer
            keep = [
                k for k, t in enumerate(latentI3d[layer][p])
                if (t % mod) != 0  # *** keep only non-coarse t ***
            ]
            latentI3d[layer][p] = [latentI3d[layer][p][k] for k in keep]
            latentJ3d[layer][p] = [latentJ3d[layer][p][k] for k in keep]
            latentK3d[layer][p] = [latentK3d[layer][p][k] for k in keep]

    return indexI3d, indexJ3d, indexK3d, latentI3d, latentJ3d, latentK3d


class MERA(HierarchyBijector):
    def __init__(self, layers,upper_layers, L, kernel_size, seq_depth,seq_len=4,prior=None):
        indexI, indexJ, latentI, latentJ = mera_indices(L, kernel_size)
        IndexISeq,IndexJSeq,IndexKSeq,latSeqI,latSeqJ,latSeqK = mera_indices_3d(int(L/8),kernel_size//2,seq_len)
        super().__init__(indexI, indexJ,latentI,latentJ,IndexISeq,IndexJSeq,IndexKSeq,latSeqI,latSeqJ,latSeqK, layers,upper_layers, seq_depth, prior)
