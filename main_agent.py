#!/usr/bin/env python3

import numpy as np
import math
import imageio
import os
import time
import traceback
from math import log, sqrt
import random
from torch.nn.utils.rnn import pad_sequence
import torch
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_

import layers
import sources
import utils
from args import args
from utils import my_log
from torch.utils.data import Dataset, DataLoader
from utils_support import my_tight_layout, plot_samples_np
import torchvision
from torchvision import datasets, transforms
from pdp_model import PDPConfig
from HMM_agent import HMMAgent
import gym
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

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
print("WHY")

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
    print("WHY")
    start_time = time.time()

    utils.init_out_dir()
    last_epoch = utils.get_last_checkpoint_step()
    # if last_epoch >= args.epoch:
    #     exit()
    if last_epoch >= 0:
        my_log(f'\nCheckpoint found: {last_epoch}\n')
    else:
        utils.clear_log()
    utils.print_args()

    flow = build_mera()
    flow.train(False)
    #torch.compile(flow)
    my_log('nparams in each RG layer: {}'.format(
        [utils.get_nparams(layer) for layer in flow.layers]))
    my_log(f'Total nparams: {utils.get_nparams(flow)}')

    # Use multiple GPUs
    if args.cuda and torch.cuda.device_count() > 1:
        flow = utils.data_parallel_wrap(flow)

    params = [x for x in flow.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  betas = (.9, .95), eps = 1e-8)
    if last_epoch >= 0:
        utils.load_checkpoint(last_epoch, flow, optimizer)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    flow = build_mera();
    flow.train(False)
    agent = HMMAgent(flow=flow, pdp_cfg=PDPConfig, device=args.device,batch_size=args.batch_size)

    optimizer = torch.optim.AdamW([p for p in flow.parameters() if p.requires_grad],
                                  lr=args.lr, betas=(.9, .95), eps=1e-8)

    # ------------------------------------------------------------------
    # 3.  Main training loop over episodes, not data‑loader
    # ------------------------------------------------------------------
    class CartEpisodeReplay(Dataset):
        def __init__(self, num_episodes, max_len=200):
            env = gym.make("CartPole-v1", render_mode="rgb_array")
            episodes = []
            for _ in range(num_episodes):
                obs_num, _ = env.reset()
                frames, acts, rews = [], [], []
                for t in range(max_len):
                    img = env.render()
                    cart_pos = obs_num[0]
                    frame = frame_transform(img)  # (C,H,W)
                    frame_log = utils.logit_transform(frame.unsqueeze(0))[0].squeeze(0)
                    action = random.randrange(env.action_space.n)
                    obs_num, _, done, trunc, _ = env.step(action)
                    reward_pole = (.418 - abs(obs_num[2])) / .418
                    reward_cart = (4.8 - abs(cart_pos)) / 4.8
                    if trunc or done:
                        reward = -1
                    elif abs(cart_pos) < .5 and abs(obs_num[2]) < .1:
                        reward = reward_pole + reward_cart
                    else:
                        reward = 0
                    frames.append(frame_log)
                    acts.append(torch.tensor(action))
                    rews.append(torch.tensor(reward, dtype=torch.float32))
                    if done or trunc:
                        break
                episodes.append((
                    torch.stack(frames, dim=0),
                    torch.stack(acts, dim=0),
                    torch.stack(rews, dim=0)
                ))
            env.close()
            self.episodes = episodes

        def __len__(self):
            return len(self.episodes)

        def __getitem__(self, i):
            return self.episodes[i]

    # ------------------------------------------------------------------
    # 2) Collate into padded minibatch of shape (B, T_max, ...)
    # ------------------------------------------------------------------
    def collate_episodes(batch):
        frames, acts, rews = zip(*batch)
        Ts = [f.size(0) for f in frames]
        Tmax = max(Ts)
        padded_frames = pad_sequence(frames, batch_first=True)
        padded_acts = pad_sequence(acts, batch_first=True, padding_value=-1)
        padded_rews = pad_sequence(rews, batch_first=True)
        mask = torch.arange(Tmax).unsqueeze(0) < torch.tensor(Ts).unsqueeze(1)
        return padded_frames, padded_acts, padded_rews, mask

    # ------------------------------------------------------------------
    # 3) Build dataset & loader
    # ------------------------------------------------------------------
    num_collect = args.collect_every
    dataset = CartEpisodeReplay(num_collect, max_len=200)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=collate_episodes,
                        pin_memory=True)

    # ------------------------------------------------------------------
    # 4) OFFLINE TRAINING on batches of episodes
    # ------------------------------------------------------------------
    epochs_offline = 1
    for epoch in range(epochs_offline):
        for batch_frames, batch_actions, batch_rewards, mask in loader:
            B, T, C, H, W = batch_frames.shape
            batch_frames = batch_frames.to(args.device)
            batch_actions = batch_actions.to(args.device)
            batch_rewards = batch_rewards.to(args.device)
            mask = mask.to(args.device)

            for t in range(T):
                valid = mask[:, t]
                if not valid.any():
                    break
                frames_t = batch_frames[valid, t]
                rewards_t = batch_rewards[valid, t]
                if frames_t.shape[0] != args.batch_size:
                    continue
                agent.infer(frames_t,
                            step=t,
                            training=True,
                            reward=rewards_t)
                torch.cuda.empty_cache()
        print(f"offline-epoch {epoch + 1}/{epochs_offline}")

    # ------------------------------------------------------------------
    # 5) ONLINE CONTROL EPISODES AND BASELINE COMPARISON
    # ------------------------------------------------------------------
    n_control = 20  # number of control episodes
    model_rewards = []
    random_rewards = []
    model_frames = []  # store frames for each episode

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    for ep in range(n_control):
        obs_num, _ = env.reset()
        ep_frames = []
        frame_img = env.render()
        ep_frames.append(frame_img)
        frame = frame_transform(frame_img).to(args.device)
        done = False
        trunc = False
        t = 0
        ep_reward = 0.0
        while not done:
            reward_pole = (.418 - abs(obs_num[2])) / .418
            reward_cart = (4.8 - abs(obs_num[0])) / 4.8
            if trunc or done:
                reward = -1
            elif abs(obs_num[0]) < .5 and abs(obs_num[2]) < .1:
                reward = reward_pole + reward_cart
            else:
                reward = 0
            ep_reward += reward

            frame_log, _ = utils.logit_transform(frame.unsqueeze(0))
            action = agent.infer(frame_log,
                                 step=t,
                                 training=False,
                                 reward=torch.tensor(reward))
            obs_num, _, done, trunc, _ = env.step(action.item())
            frame_img = env.render()
            ep_frames.append(frame_img)
            frame = frame_transform(frame_img).to(args.device)
            t += 1

        model_rewards.append(ep_reward)
        model_frames.append(ep_frames)
        print(f"Model-Ep {ep:03d} | steps={t:3d} | total-reward={ep_reward:8.2f}")

    # RANDOM BASELINE
    for ep in range(n_control):
        obs_num, _ = env.reset()
        done = False
        trunc = False
        t = 0
        ep_reward = 0.0
        while not done:
            action = random.randrange(env.action_space.n)
            obs_num, _, done, trunc, _ = env.step(action)
            reward_pole = (.418 - abs(obs_num[2])) / .418
            reward_cart = (4.8 - abs(obs_num[0])) / 4.8
            if trunc or done:
                reward = -1
            elif abs(obs_num[0]) < .5 and abs(obs_num[2]) < .1:
                reward = reward_pole + reward_cart
            else:
                reward = 0
            ep_reward += reward
            t += 1
        random_rewards.append(ep_reward)
        print(f"Random-Ep {ep:03d} | steps={t:3d} | total-reward={ep_reward:8.2f}")

    env.close()

    # ------------------------------------------------------------------
    # 6) Performance Plot and Save
    # ------------------------------------------------------------------
    episodes = list(range(1, n_control + 1))
    plt.figure()
    plt.plot(episodes, model_rewards, label='Model')
    plt.plot(episodes, random_rewards, label='Random')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison.png')
    print("Saved performance plot to 'performance_comparison.png'")
    plt.close()

    # ------------------------------------------------------------------
    # 7) Save Best Episode as Looping GIF
    # ------------------------------------------------------------------
    best_idx = int(torch.tensor(model_rewards).argmax().item())
    best_frames = model_frames[best_idx]
    gif_path = f'best_episode_{best_idx + 1}.gif'
    # create a looping GIF (loop=0)
    imageio.mimsave(gif_path, best_frames, fps=30, loop=0)
    print(f"Saved best episode ({best_idx + 1}) as looping GIF to '{gif_path}'")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()

