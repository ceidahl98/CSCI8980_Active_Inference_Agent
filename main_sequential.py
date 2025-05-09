#!/usr/bin/env python3

import numpy as np
import math
import imageio
import os
import time
import traceback
from math import log, sqrt
import random

import torch
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_

import layers
import sources
import utils
from args import args
from utils import my_log
from utils_support import my_tight_layout, plot_samples_np
import torchvision
from torchvision import datasets, transforms
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
import gym

torch.backends.cudnn.benchmark = True


def get_prior(temperature=1):
    if args.prior == 'gaussian':
        prior = sources.Gaussian([args.nchannels, args.L, args.L],
                                 scale=temperature)
    elif args.prior == 'laplace':
        # Set scale = 1/sqrt(2) to make var = 1
        prior = sources.Laplace([args.nchannels, args.L, args.L],
                                scale=temperature / sqrt(2))
    elif args.prior =='poisson':
        prior = sources.Poisson([args.nchannels, args.L, args.L],
                                scale=temperature)
    elif args.prior == 'gamma':
        prior = sources.Gamma([args.nchannels, args.L, args.L],
                              args.concentration,
                              args.rate,
                              scale=temperature)
    else:
        raise ValueError(f'Unknown prior: {args.prior}')
    prior = prior.to(args.device)
    return prior


def build_rnvp(nchannels, kernel_size, nlayers, nresblocks, nmlp, nhidden):
    core_size = nchannels * kernel_size**2
    widths = [core_size] + [nhidden] * nmlp + [core_size]
    net = layers.RNVP(
        [
            layers.ResNetReshape(
                nresblocks,
                widths,
                final_scale=True,
                final_tanh=True,
                num_heads=4,
                final_relu=False
            ) for _ in range(nlayers)
        ],
        [
            layers.ResNetReshape(
                nresblocks,
                widths,
                final_scale=True,
                final_tanh=False,
                num_heads=4,
                final_relu=False
            ) for _ in range(nlayers)
        ],
        nchannels,
        kernel_size,
    )
    return net

# def build_rnvp(nchannels, kernel_size, nlayers, nresblocks, nmlp, nhidden):
#     core_size = nchannels * kernel_size**2
#     widths = [core_size] + [nhidden] * nmlp + [core_size]
#     print(widths)
#     net = layers.RNVP(
#         [
#             layers.ResNetReshapeTransformer(
#                 kernel_size,
#                 nchannels,
#                 widths[1],
#                 4,
#                 nlayers,
#                 i,
#                 final_relu=False
#             ) for i in range(nlayers)
#         ],
#         [
#             layers.ResNetReshapeTransformer(
#                 kernel_size,
#                 nchannels,
#                 widths[1],
#                 4,
#                 nlayers,
#                 i,
#                 final_relu=True
#             ) for i in range(nlayers)
#         ],
#         nchannels,
#         kernel_size,
#     )
#     return net




def build_arflow(nchannels, kernel_size, nlayers, nresblocks, nmlp, nhidden):
    assert nhidden % kernel_size**2 == 0
    channels = [nchannels] + [nhidden // kernel_size**2] * nmlp + [nchannels]
    width = kernel_size**2
    net = layers.ARFlowReshape(
        [
            layers.MaskedResNet(
                nresblocks,
                channels,
                width,
                final_scale=True,
                final_tanh=True,
                final_relu=False,
            ) for _ in range(nlayers)
        ],
        [
            layers.MaskedResNet(
                nresblocks,
                channels,
                width,
                final_scale=True,
                final_tanh=False,
                final_relu=False,

            ) for _ in range(nlayers)
        ],
    )
    return net


def build_mera():
    prior = get_prior()

    _layers = []
    upper_layers = []
    for i in range(args.depth):
        if args.subnet == 'rnvp':
            _layers.append(
                build_rnvp(
                    args.nchannels,
                    args.kernel_size,
                    args.nlayers_list[i],
                    args.nresblocks_list[i],
                    args.nmlp_list[i],
                    args.nhidden_list[i],
                ))
        elif args.subnet == 'ar':
            _layers.append(
                build_arflow(
                    args.nchannels,
                    args.kernel_size,
                    args.nlayers_list[i],
                    args.nresblocks_list[i],
                    args.nmlp_list[i],
                    args.nhidden_list[i],
                ))


        else:
            raise ValueError(f'Unknown subnet: {args.subnet}')

    for i in range(4):

        upper_layers.append(
            build_rnvp(
                args.nchannels*2,
                args.kernel_size,
                args.nlayers_list[-1],
                args.nresblocks_list[-1],
                args.nmlp_list[-1],
                args.nhidden_list[-1],
            ))


    flow = layers.MERA(_layers,upper_layers,args.L, args.kernel_size,args.seq_depth, prior=prior)
    flow = flow.to(args.device)

    return flow


def do_plot(flow, epoch_idx):
    flow.train(False)

    sample, _ = flow.sample(args.batch_size // args.device_count, device=args.device)
    my_log('plot min {:.3g} max {:.3g} mean {:.3g} std {:.3g}'.format(
        sample.min().item(),
        sample.max().item(),
        sample.mean().item(),
        sample.std().item(),
    ))


    sample, _ = utils.logit_transform(sample, inverse=True)
    sample = torch.clamp(sample, 0, 1)
    frames = 4**(args.seq_depth-1)
    C = args.nchannels
    L = args.L
    sample = sample.detach().cpu().numpy().reshape(-1, frames, C, L, L)  # shape: (N, 4, 1, 32, 32)

    N, T, _, H, W = sample.shape  # N=number of sequences, T=timesteps

    grid_size = math.ceil(math.sqrt(N))
    frame_images = []

    for t in range(T):  # for each timestep
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))

        for idx in range(grid_size * grid_size):
            row, col = divmod(idx, grid_size)
            ax = axs[row, col] if grid_size > 1 else axs

            if idx < N:
                ax.imshow(sample[idx, t, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis("off")

        fig.tight_layout()
        fig.canvas.draw()

        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame_images.append(img)
        plt.close(fig)

    gif_dir = f'{args.plot_filename}/gifs_epoch{epoch_idx}'
    os.makedirs(gif_dir, exist_ok=True)
    gif_path = os.path.join(gif_dir, f"sequence_grid_epoch{epoch_idx}.gif")
    imageio.mimsave(gif_path, frame_images, fps=4, loop=0)

    flow.train(True)


frame_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32,32)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
])


def transform_video(video):
    """
    Applies a per-frame transformation to a video clip.

    Args:
        video: A video clip represented as a numpy array or tensor of shape (T, H, W) or (T, H, W, C)

    Returns:
        A tensor of shape (T, C, H_new, W_new) after applying the frame_transform to each frame.
    """
    transformed_frames = []
    for frame in video:
        transformed_frame = frame_transform(frame)
        transformed_frames.append(transformed_frame)
    # Stack frames along a new time dimension.
    return torch.stack(transformed_frames)

def main():
    start_time = time.time()

    utils.init_out_dir()
    last_epoch = utils.get_last_checkpoint_step()
    if last_epoch >= args.epoch:
        return
    if last_epoch >= 0:
        my_log(f'\nCheckpoint found: {last_epoch}\n')
    else:
        utils.clear_log()
    utils.print_args()

    flow = build_mera().to(args.device)
    flow.train(True)
    if args.cuda and torch.cuda.device_count() > 1:
        flow = utils.data_parallel_wrap(flow)
    params = [p for p in flow.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(.9,.95), eps=1e-8)
    if last_epoch >= 0:
        utils.load_checkpoint(last_epoch, flow, optimizer)

    my_log(f'Init time: {time.time()-start_time:.3f}s')
    my_log('Collecting CartPole episodes…')

    env = gym.make('CartPole-v1', render_mode='rgb_array')
    episodes_buffer = []
    seq_len       = 4
    collect_every = args.collect_every      # e.g. 10
    total_eps     = args.total_episodes     # e.g. 200

    for ep in range(total_eps):
        frames = []
        done   = False
        _obs, _ = env.reset()

        # collect one full episode as raw RGB frames
        while not done:
            img = env.render()  # shape (H, W, C)
            frames.append(img)
            action, = env.action_space.sample(),

            obs, reward, done, info,_ = env.step(action)

        # convert to (T, C, H, W)
        ep_raw = np.stack(frames, axis=0)


        ep_transformed = transform_video(ep_raw)  # shape (T, C, Hn, Wn)

        episodes_buffer.append(ep_transformed)  # store tensor

        # once we've collected enough episodes, build sequences & train
        if (ep+1) % collect_every == 0:
            # 1) break each episode into back-to-back seq_len chunks
            # 1) break each episode into sliding windows of length seq_len, stride=1
            seqs = []
            for ep_arr in episodes_buffer:
                T = ep_arr.shape[0]
                if T < seq_len:
                    continue
                # for each possible start i, grab frames [i : i+seq_len]
                for i in range(0, T - seq_len + 1):
                    seqs.append(ep_arr[i: i + seq_len])

            # 2) shuffle all sequences
            random.shuffle(seqs)

            # 3) train over mini-batches
            print(range(0, len(seqs), args.batch_size))
            for i in range(0, len(seqs), args.batch_size):
                batch = torch.stack(seqs[i: i + args.batch_size], dim=0).to(args.device)
                # batch: (B, seq_len, C, Hn, Wn)
                B, T, C, Hn, Wn = batch.shape

                # flatten for logit_transform
                x_flat = batch.view(B * T, C, Hn, Wn)
                x_flat, ldj_logit = utils.logit_transform(x_flat)
                x = x_flat.view(B, T, C, Hn, Wn)

                optimizer.zero_grad()

                lower_p, lower_ldj, upper_ps, upper_ldjs = flow.log_prob(x)

                loss = - (lower_p + lower_ldj + ldj_logit)/(args.nchannels * args.L ** 2)
                loss = loss.mean()
                for p, l in zip(upper_ps, upper_ldjs):
                    loss = loss - ((p + l)/(args.nchannels*4*(args.L/8)**2)).mean()
                loss = loss

                loss.backward()
                if args.clip_grad:
                    clip_grad_norm_(params, args.clip_grad)
                optimizer.step()

                if args.print_step and i % args.print_step == 0:
                    bit_per_dim = (loss.item() + log(256)) / log(2)
                    my_log(
                        'epoch {} batch {} bpp {:.8g} loss {:.8g} +- {:.8g} time {:.3f}'
                        .format(
                            ep,
                            i,
                            bit_per_dim,
                            loss.item(),
                            loss.std().item(),
                            time.time() - start_time,
                        ))

            # clear buffer and repeat
            episodes_buffer.clear()



            # print("Gradients for model parameters:")
            # for name, param in flow.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.mean().item()}")
            #     else:
            #         print(f"{name} has no GRADIENT!")



            # if counter == 10:
            #     break
            # counter+=1
            if (args.out_filename and args.save_epoch
                    and ep % args.save_epoch == 0):
                state = {
                    'flow': flow.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, f'{args.out_filename}_save/{ep}.state')

                last_epoch = ep+1 - args.save_epoch
                if (last_epoch > 0 and args.keep_epoch
                        and last_epoch % args.keep_epoch != 0):
                    os.remove(f'{args.out_filename}_save/{last_epoch}.state')

                if (args.plot_filename and args.plot_epoch
                        and ep % args.plot_epoch == 0):
                    with torch.no_grad():
                        do_plot(flow, ep)





if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
