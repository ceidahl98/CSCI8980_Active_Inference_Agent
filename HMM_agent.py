from math import log

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from pdp_model import PoissonDirichlet
from contextlib import contextmanager

class HMMAgent(nn.Module):
    def __init__(self, flow, pdp_cfg,action_space=2,prior_init_val=1, horizon=4,batch_size=16, device="cuda"):
        super().__init__()
        self.flow       = flow.to(device)
        self.batch_size = batch_size
        self.dirichlet_init_val = prior_init_val
        num_spatial_latents = len(flow.layers)//2
        num_temporal_latents = len(flow.upper_layers)//2
        spatial_features_full = int(flow.patch_size**2 * flow.channels)
        spatial_features_partial = int(spatial_features_full*.75)
        temporal_features_full = int(flow.patch_size**2*2)
        temporal_features_partial = int(temporal_features_full*.5)
        #init pdp for spatial latents
        self.total_layers = num_spatial_latents + num_temporal_latents
        self.spatial_PDPs = nn.ModuleList().cpu()
        for i in range(num_spatial_latents):
            self.spatial_PDPs.append(
                PoissonDirichlet(
                    prior_mu = pdp_cfg.prior_mu,
                    prior_sigma = pdp_cfg.prior_sigma,
                    a = pdp_cfg.a,
                    b = pdp_cfg.b,
                    max_clusters=pdp_cfg.max_clusters,
                    num_iterations=pdp_cfg.num_iterations,
                    num_features=spatial_features_partial if i != (num_spatial_latents-1) else spatial_features_full,
                    merge_threshold=pdp_cfg.merge_threshold,
                    prior_kappa=pdp_cfg.prior_kappa,
                    prior_alpha=pdp_cfg.prior_alpha,
                    prior_beta=pdp_cfg.prior_beta,
                    sep_quantile=pdp_cfg.sep_quantile,
                    density_quantile=pdp_cfg.sep_density,
                    ema_decay = pdp_cfg.ema_decay,
                    num_candidates = pdp_cfg.num_candidates

                ).cpu()
            )

        self.temporal_PDPs = nn.ModuleList().to('cpu')
        for i in range(num_temporal_latents):
            self.temporal_PDPs.append(
                PoissonDirichlet(
                    prior_mu=pdp_cfg.prior_mu,
                    prior_sigma=pdp_cfg.prior_sigma,
                    a=pdp_cfg.a,
                    b=pdp_cfg.b,
                    max_clusters=pdp_cfg.max_clusters,
                    num_iterations=pdp_cfg.num_iterations,
                    num_features=temporal_features_partial if i != (num_temporal_latents-1) else temporal_features_full,
                    merge_threshold=pdp_cfg.merge_threshold,
                    prior_kappa=pdp_cfg.prior_kappa,
                    prior_alpha=pdp_cfg.prior_alpha,
                    prior_beta=pdp_cfg.prior_beta,
                    sep_quantile=pdp_cfg.sep_quantile,
                    density_quantile=pdp_cfg.sep_density,
                    ema_decay=pdp_cfg.ema_decay,
                    num_candidates=pdp_cfg.num_candidates

                ).cpu()
            )



        num_paths_spatial = [action_space for _ in range(num_spatial_latents)]
        num_paths_temporal = [action_space if i is 0 else 4 for i in range(num_temporal_latents)]
        self.num_paths = num_paths_spatial+num_paths_temporal
        self.A, self.B, self.C, self.D, self.E = None,None, None, None, None  # lazy init
        self.states = None
        self.current_policy = None
        self.previous_policy = None

        self.timescales = [1 for _ in range(num_spatial_latents)] + [2*(i+1) for i in range(num_temporal_latents)]
        self.num_children = [4 if i is not 0 else 1 for i in range(num_spatial_latents)] + [2 for _ in range(num_temporal_latents)]

        self.horizon = horizon
        self.device  = device
        self.latent_buffer = []
        self.D_message_buffer = [[] for _ in range(self.total_layers)]
        self.E_messages_buffer = [[] for _ in range(self.total_layers)]
        self.free_energy = [0 for _ in range(self.total_layers)]
        self.H_top = 1
        self.k_down = [2,2]
        self.temporal_idx = [5,4,3,2,1,0]
        self.free_energy_acc = [0 for _ in range(self.total_layers)]
        n_temporal = len(self.temporal_idx)
        n_spatial = self.total_layers - n_temporal
        self.delay_counter = 0
        self.layer_order = self.temporal_idx + [
            idx for idx in range(self.total_layers)
            if idx not in set(self.temporal_idx)
        ]
        self.layer_order=self.temporal_idx
        self.layer_types = ['temporal'] * n_temporal + ['spatial'] * n_spatial

    # ----------------------------------------------------------------------

    # map image space to latent space with RG flow model
    def encode(self, x):

        spatial_latents = self.flow.forward_spatial(x)
        self.latent_buffer.append(spatial_latents[-1].detach().cpu())
        if len(self.latent_buffer) >4:
            self.latent_buffer = self.latent_buffer[-4:]
        self.latent_buffer = [latent.to(self.device) for latent in self.latent_buffer]
        temporal_latents = self.flow.forward_temporal(self.latent_buffer)

        return spatial_latents,temporal_latents                                # list[Tensor] , list[Tensor

    #pdp fits to current data and returns observation probability vectors for each level
    def observe(self,x,training=True):
        spatial_latents,temporal_latents = self.encode(x)
        if training is True:
            spatial_obs = [pdp.fit(latent)[-1] for pdp,latent in zip(self.spatial_PDPs,spatial_latents)]
            temporal_obs = [pdp.fit(latent)[-1] if latent is not None else None for pdp,latent in zip(self.temporal_PDPs,temporal_latents)]
        else:
            spatial_obs = [pdp.assign_clusters(latent)[-1]for pdp,latent in zip(self.spatial_PDPs,spatial_latents)]
            temporal_obs = [pdp.assign_clusters(latent)[-1] if latent is not None else None for pdp, latent in zip(self.temporal_PDPs, temporal_latents)]
        self._update_prior_shapes() #add new clusters found by PDP

        return spatial_obs,temporal_obs

    def normalize_dist(self,x):
        x = x / x.sum(-1, keepdim=True)
        return x

    def inflate_state(self,x_prev, new_S):

        if x_prev.shape[-1] == new_S:  # no growth
            return x_prev

        pad_shape = list(x_prev.shape)
        pad_shape[-1] = new_S - x_prev.shape[-1]

        pad = x_prev.new_zeros(*pad_shape)  # zeros for never‑seen clusters
        return torch.cat([x_prev, pad], dim=-1)

    def inflate_policy(self, pi: torch.Tensor, U_new: int, fill="zeros"):

        U_old = pi.shape[-1]
        if U_new == U_old:
            return pi

        shape_pad = list(pi.shape)
        shape_pad[-1] = U_new - U_old

        if fill == "uniform":
            pad = pi.new_full(shape_pad, 1.0 / U_new)  # tiny but non‑zero
        else:  # "zeros"
            pad = pi.new_zeros(*shape_pad)

        pi_expanded = torch.cat([pi, pad], dim=-1)
        # renormalise so it sums to 1 along the last axis
        return pi_expanded / pi_expanded.sum(dim=-1, keepdim=True)
    def init_child_from_D(self,D, eps=1e-16):
        D = D.concentration
        cond_prob = D / (D.sum(dim=0, keepdim=True) + eps)  # (S_child, S_parent)

        # 2. Uniform prior over parents  ⇒  simple column‑mean
        Q_child = cond_prob.mean(dim=1)  # (S_child,)
        return Q_child

    def digamma(self,a):
        a0 = a.concentration.sum(-1, keepdim=True)  # shape (..., 1)
        return torch.digamma(a.concentration) - torch.digamma(a0)
    def _get_messages(self,obs=None,layer_idx=0,tick=None, layer_type = 'spatial',plan_gamma=None): #infer state for a given layer and calculate free energy

        num_children = self.num_children[layer_idx+1] if layer_idx is not self.total_layers-1 else 2
        tick_up = tick[layer_idx+1] if layer_idx is not self.total_layers-1 else True
        obs = obs.to(self.device) if obs is not None else None
        current_state =  self.states[layer_idx].to(self.device)
        previous_state = self.previous_state[layer_idx].to(self.device) if self.previous_state[layer_idx] is not None else None
        previous_policy = self.previous_policy[layer_idx].to(self.device) if self.previous_policy[layer_idx] is not None else None
        current_policy = self.current_policy[layer_idx].to(self.device)
        lower_state = None if layer_idx is 0 else self.states[layer_idx-1].to(self.device)
        upper_state = None if layer_idx is self.total_layers-1 else self.states[layer_idx+1].to(self.device)
        lower_policy = None if layer_idx is 0 else self.current_policy[layer_idx-1].to(self.device)
        if layer_type is 'temporal' and self.previous_state[layer_idx-1] is not None:
            lower_state_previous = self.previous_state[layer_idx-1].to(self.device)
            lower_state_previous = lower_state_previous
            lower_state = torch.stack([lower_state_previous,lower_state],dim=1)
            lower_policy_previous = self.previous_policy[layer_idx-1].to(self.device)
            lower_policy_previous = lower_policy_previous
            lower_policy = torch.stack([lower_policy_previous,lower_policy],dim=1)
        else:
            lower_state_previous = None
            lower_policy_previous = None



        A = dist.Dirichlet(self.A[layer_idx].to(self.device))
        B = dist.Dirichlet(self.B[layer_idx].to(self.device))
        C = dist.Dirichlet(self.C[layer_idx].to(self.device))
        U = dist.Dirichlet(self.U[layer_idx].to(self.device))

        D_upper = dist.Dirichlet(self.D[layer_idx].to(self.device))
        D_lower = dist.Dirichlet(self.D[layer_idx-1].to(self.device)) if layer_idx is not 0 else None
        E_upper = dist.Dirichlet(self.E[layer_idx].to(self.device))
        E_lower = dist.Dirichlet(self.E[layer_idx-1].to(self.device)) if layer_idx is not 0 else None


        #prior_policy = self.normalize_dist(U.concentration)#.unsqueeze(0).expand(current_state.shape[0],-1,-1)

        u_A_up = (obs * current_state) @ self.digamma(A) if obs is not None else None  #(B x num_states)

        if tick_up:
            u_D_down = (self.digamma(D_upper) @ upper_state.T).view(self.batch_size,-1,1,current_state.shape[-1]) if upper_state is not None else self.digamma(D_upper).sum(dim=1).view(1,1,1,-1)

            u_E_down= (upper_state @ self.digamma(E_upper).T).view(self.batch_size,-1,1,current_policy.shape[-1]) if upper_state is not None else self.digamma(E_upper).sum(dim=1).view(1,1,1,-1)

        else:
            u_D_down = None
            u_E_down = None

        if layer_type is 'spatial':

            u_D_up = torch.matmul(lower_state.view(-1,4,lower_state.shape[-1]),self.digamma(D_lower)).sum(dim=1) * current_state if lower_state is not None else None

            u_E_up = torch.matmul(lower_policy.view(-1,4,lower_policy.shape[-1]), self.digamma(E_lower)).sum(dim=1) * current_state if lower_state is not None else None

        if layer_type is 'temporal':
            if plan_gamma is None:
                u_D_up = torch.matmul(lower_state,self.digamma(D_lower)).sum(dim=1) * current_state
                u_E_up = torch.matmul(lower_policy, self.digamma(E_lower)).sum(dim=1) * current_state
            else:
                u_D_up = None
                u_E_up = None


        H = (self.normalize_dist(A.concentration) * self.digamma(A)).sum(dim=0)

        gamma = plan_gamma if plan_gamma is not None else 1.0
        efe,Q_u = self.one_step_efe(current_state,q_u=current_policy,B=B.concentration,tilde_A=self.normalize_dist(A.concentration),H_A=H,log_c=self.digamma(C),gamma=gamma)
        #efe = (next_obs_pred * (torch.log(next_obs_pred+1e-12) - self.digamma(C).T.unsqueeze(0)) - next_state_pred @ H.unsqueeze(0).T).sum(dim=-1)
        c_efe = self.free_energy_acc[layer_idx]

        self.free_energy_acc[layer_idx] = c_efe + efe if self.free_energy_acc[layer_idx] is not None else efe

        # if plan_gamma is not None:
        #     u_C = torch.log(torch.softmax(-gamma * efe, dim=-1)+ 1e-12)
        # else:
        #
        #     u_C = (previous_policy.unsqueeze(-1) * (self.digamma(U))).sum(-1) if previous_policy is not None else None
        u_C = torch.log(torch.softmax(-gamma * efe, dim=-1) + 1e-12)

        u_B_right = torch.einsum('ns,nu,sku->nk',previous_state,previous_policy,self.digamma(B)) if previous_state is not None else None

        u_B_up = torch.einsum('ns,nk,sku->nu',current_state,previous_state,self.digamma(B)) if previous_state is not None else None

        if layer_idx == len(self.spatial_PDPs)-1 or layer_type is 'temporal':
            num_children = 1
        u_A_up = u_A_up.view(self.batch_size,-1,num_children,current_state.shape[-1]) if u_A_up is not None else None
        u_B_right = u_B_right.view(self.batch_size,-1,num_children,current_state.shape[-1]) if u_B_right is not None else None
        u_B_up = u_B_up.view(self.batch_size,-1,num_children,current_policy.shape[-1]) if u_B_up is not None else None
        u_D_up = u_D_up.view(self.batch_size,-1,num_children,current_state.shape[-1]) if u_D_up is not None else None
        if layer_idx is not self.total_layers-1:
            u_D_down = u_D_down.view(self.batch_size,-1,1,current_state.shape[-1]) if u_D_down is not None else None
            u_E_down = u_E_down.view(self.batch_size, -1, 1, current_policy.shape[-1]) if u_E_down is not None else None
        else:
            u_D_down = u_D_down.view(1, -1, 1, current_state.shape[-1]) if u_D_down is not None else None
            u_E_down = u_E_down.view(1, -1, 1, current_policy.shape[-1]) if u_E_down is not None else None


        u_E_up = u_E_up.view(self.batch_size,-1,num_children,current_state.shape[-1]) if u_E_up is not None else None
        if plan_gamma is not None:
            u_D_up = None
            u_E_up = None


        A_messages = [u_A_up]
        B_messages = [u_B_right] #will have to add left and up messages if training offline
        D_messages = [u_D_up,u_D_down]
        E_messages = [u_E_up,u_E_down]





        state_messages = A_messages + [u_B_right] + D_messages + [u_E_up]
        total_state_message = None
        for m in state_messages:
            if m is None:
                continue
            else:

                total_state_message = m if total_state_message is None else total_state_message + m
        u_C = u_C.view(self.batch_size,-1,num_children,u_C.shape[-1]) if u_C is not None else None
        policy_messages = [u_C, u_E_down,u_B_up]
        total_policy_message = None
        for p in policy_messages:
            if p is None:
                continue
            else:
                total_policy_message = p if total_policy_message is None else total_policy_message+p
        #update states

        self.states[layer_idx] = torch.softmax(total_state_message.view(-1,current_state.shape[-1]),dim=-1).detach().cpu()

        self.current_policy[layer_idx] = torch.softmax(total_policy_message.view(-1,current_policy.shape[-1]),dim=-1).detach().cpu()
        current_state = self.states[layer_idx].to(self.device)
        current_policy = self.current_policy[layer_idx].to(self.device)
        if plan_gamma is None:
            if previous_state is None:
                state_free_energy = (current_state * (torch.log(current_state+1e-12)-total_state_message.view(-1,current_state.shape[-1]))).sum()
            else:
                state_free_energy = (current_state * (torch.log(current_state+1e-12) - u_B_right.view(-1,current_state.shape[-1])) - current_state * u_A_up.view(-1,current_state.shape[-1])).sum()
            path_free_energy = (current_policy * (torch.log(current_policy + 1e-12)-total_policy_message.view(-1,current_policy.shape[-1]))).sum()
            self.free_energy[layer_idx] = (state_free_energy + path_free_energy).detach().cpu()




    def _init_nodes(self):
        states_spatial = [torch.softmax(self.init_child_from_D(dist.Dirichlet(self.D[i])),dim=-1).unsqueeze(0).expand((64*self.batch_size)//4**i,-1) for i in range(len(self.spatial_PDPs))] #set to values by D posterior
        states_temporal = [torch.softmax(self.init_child_from_D(dist.Dirichlet(self.D[i+len(self.spatial_PDPs)])),dim=-1).unsqueeze(0).expand((self.batch_size),-1) for i in range(len(self.temporal_PDPs))]

        self.states = states_spatial + states_temporal

        self.previous_state = [None for _ in range(self.total_layers)]
        policy_spatial = [torch.softmax(self.init_child_from_D(dist.Dirichlet(self.E[i])),dim=-1).unsqueeze(0).expand((64*self.batch_size)//4**i,-1) for i in range(len(self.spatial_PDPs))]
        policy_temporal = [torch.softmax(self.init_child_from_D(dist.Dirichlet(self.E[i+len(self.spatial_PDPs)])),dim=-1).unsqueeze(0).expand((self.batch_size),-1) for i in range(len(self.temporal_PDPs))]
        self.current_policy = policy_spatial + policy_temporal

        self.previous_policy = [None for _ in range(self.total_layers)]


    def infer(self, data,step,episode_start=True,max_sweeps=10,tol=1e-6,training=False,reward=None):
        self.batch_size = data.shape[0]
        self.free_energy_acc = [None for _ in range(self.total_layers)]
        ticks = [step+1 % scale == 0 for scale in self.timescales]

        if step!=0:
            episode_start=False


        if episode_start is True:
            self.latent_buffer.clear()

        spatial_obs, temporal_obs = self.observe(data,training)

        if episode_start is True:
            self._init_nodes() #initialize states and policies based on priors

        layer_spec = ([(o, 'spatial') for o in spatial_obs] +
                      [(o, 'temporal') for o in temporal_obs])

        prev_global_FE = float('inf')
        for sweep in range(max_sweeps):


            for idx, (obs, ltype) in enumerate(layer_spec):
                if obs is None:
                    continue
                self._get_messages(obs, idx, ticks, layer_type=ltype)


            global_FE = sum(self.free_energy)  # scalar

            if prev_global_FE - global_FE < tol * abs(global_FE):
                break
            prev_global_FE = global_FE
        else:
            print(f"[warn] hierarchy hit max_sweeps={max_sweeps} without converging")

        print(global_FE,"GLOBAL_FE")
        if training is False:
            self.update_priors(spatial_obs + temporal_obs, reward=reward, eta=1,alpha_gate=512)
            with self.saved_beliefs():
                self.rollout_predictive_trajectory(horizon=3)
                for i, pdp in enumerate(self.spatial_PDPs):
                    print(pdp.num_clusters,"Spatial"+str(i))
                for i,pdp in enumerate(self.temporal_PDPs):
                    print(pdp.num_clusters,"Temporal"+str(i))

                acc_free_energy = None

                for fe in self.free_energy_acc[:4]:
                    buffer = fe.view(self.batch_size,-1,2)
                    blocks = buffer.shape[1]
                    acc_free_energy = acc_free_energy + buffer.sum(dim=1)/blocks if acc_free_energy is not None else buffer.sum(dim=1)/blocks
                #acc_free_energy_motor = self.free_energy_acc[0]+self.free_energy_acc[1]+self.free_energy_acc[2]+self.free_energy_acc[3]
                print(acc_free_energy,"FREE")
                action = torch.argmin(torch.softmax(-acc_free_energy,dim=-1),dim=-1)
        else:
            if self.delay_counter<10:
                reward = None
            self.update_priors(spatial_obs + temporal_obs, reward=reward,eta=1,alpha_gate=512)
            action=None


        self.delay_counter+=1
        #roll model forward one timestep
        self.previous_state = [q.detach().cpu() for q in self.states if q is not None]
        self.previous_policy = [u.detach().cpu() for u in self.current_policy if u is not None]
        q_pred_state = []
        for qs,qu,B in zip(self.states,self.current_policy,self.B):
            weighted_T = self.normalize_dist(torch.einsum('snu,bu->bns',B,qu))
            q_pred_state.append(torch.einsum('bns,bs->bn',weighted_T,qs)
            )
        q_pred_policy = [qu @ self.normalize_dist(u) for qu, u in zip(self.current_policy, self.U)]
        self.states = [q.detach().cpu() for q in q_pred_state]
        self.current_policy = [u.detach().cpu() for u in q_pred_policy]

        return action


    def _update_prior_shapes(self):

        spatial_shapes  = [pdp.num_clusters for pdp in self.spatial_PDPs]
        temporal_shapes = [pdp.num_clusters for pdp in self.temporal_PDPs]
        shapes          = spatial_shapes + temporal_shapes
        L = len(shapes)
        init_val = self.dirichlet_init_val


        if self.A is None:
            return self._init_priors()


        for l in range(L):
            S = shapes[l]
            U = self.num_paths[l]

            A = self.A[l].detach()
            oldS = A.shape[0]
            if S != oldS:
                # rows
                if S > oldS:
                    new_rows = torch.ones((S-oldS, oldS), device=A.device) * init_val
                    A = torch.cat([A, new_rows], dim=0)        # (S×oldS)
                else:
                    pass
                # cols
                if S > oldS:
                    new_cols = torch.ones((S, S-oldS), device=A.device) * init_val
                    A = torch.cat([A, new_cols], dim=1)        # (S×S)
                else:
                    pass
                self.A[l] = nn.Parameter(A,requires_grad=False)


            B = self.B[l].detach()
            oldS, _, oldU = B.shape

            # a) grow state dims
            if S != oldS:
                if S > oldS:
                    # axis=0
                    new0 = torch.ones((S-oldS, oldS, oldU), device=B.device) * init_val
                    B = torch.cat([B, new0], dim=0)         # (S×oldS×oldU)
                    # axis=1
                    new1 = torch.ones((S, S-oldS, oldU), device=B.device) * init_val
                    B = torch.cat([B, new1], dim=1)         # (S×S×oldU)
                else:
                    pass

            # b) grow action dim
            if U != oldU:
                if U > oldU:
                    new2 = torch.ones((S, S, U-oldU), device=B.device) * init_val
                    B  = torch.cat([B, new2], dim=2)       # (S×S×U)
                else:
                    pass

            self.B[l] = nn.Parameter(B,requires_grad=False)




            Cmat  = self.U[l].detach()
            oldU2 = Cmat.shape[0]
            if U != oldU2:
                # rows
                if U > oldU2:
                    new_r = torch.ones((U-oldU2, oldU2), device=Cmat.device) * init_val
                    Cmat = torch.cat([Cmat, new_r], dim=0)   # (U×oldU2)
                else:
                    Cmat = Cmat[:U, :oldU2]
                # cols
                if U > oldU2:
                    new_c = torch.ones((U, U-oldU2), device=Cmat.device) * init_val
                    Cmat = torch.cat([Cmat, new_c], dim=1)   # (U×U)
                else:
                    pass
                self.U[l] = nn.Parameter(Cmat,requires_grad=False)

            Cvec = self.C[l].detach()  # (old_O, 1)
            old_O = Cvec.shape[0]
            new_O = S  # outcomes grow/shrink with child‑states

            if new_O != old_O:
                if new_O > old_O:  # grow --> pad NEW rows
                    pad = torch.full(
                        (new_O - old_O, 1),
                        init_val,
                        device=Cvec.device
                    )
                    Cvec = torch.cat([Cvec, pad], dim=0)
                else:  # shrink --> keep leading slice
                    pass

            self.C[l] = nn.Parameter(Cvec,requires_grad=False)  # keep shape (new_O, 1)


            # D: p(s_child|s_parent)
            S = shapes[l]  # # of child‐states at layer l

            # --- D: p(s_child | s_parent), shape = (children, parents) ---
            Dmat = self.D[l].detach()
            old_children, old_parents = Dmat.shape

            # decide new size
            if l != L - 1:
                new_children = S
                new_parents = shapes[l + 1]
            else:
                new_children = S
                new_parents = 1

            # grow/shrink rows = children
            if new_children != old_children:
                if new_children > old_children:
                    rows = torch.full(
                        (new_children - old_children, old_parents),
                        init_val,
                        device=Dmat.device
                    )
                    Dmat = torch.cat([Dmat, rows], dim=0)
                else:
                    pass

            # grow/shrink cols = parents
            if new_parents != old_parents:
                if new_parents > old_parents:
                    cols = torch.full(
                        (new_children, new_parents - old_parents),
                        init_val,
                        device=Dmat.device
                    )
                    Dmat = torch.cat([Dmat, cols], dim=1)
                else:
                    pass

            self.D[l] = nn.Parameter(Dmat,requires_grad=False)

            Emat = self.E[l].detach()
            old_actions, old_parents2 = Emat.shape


            if l != L - 1:
                new_actions = U
                new_parents2 = shapes[l + 1]
            else:
                new_actions = U
                new_parents2 = 1

            # grow/shrink rows = actions
            if new_actions != old_actions:
                if new_actions > old_actions:
                    rows2 = torch.full(
                        (new_actions - old_actions, old_parents2),
                        init_val,
                        device=Emat.device
                    )
                    Emat = torch.cat([Emat, rows2], dim=0)
                else:
                    pass

            # grow/shrink cols = parents
            if new_parents2 != old_parents2:
                if new_parents2 > old_parents2:
                    cols2 = torch.full(
                        (new_actions, new_parents2 - old_parents2),
                        init_val,
                        device=Emat.device
                    )
                    Emat = torch.cat([Emat, cols2], dim=1)
                else:
                    pass

            self.E[l] = nn.Parameter(Emat,requires_grad=False)
            if l < L-1:

                J = self.total_joint_su[l].detach()
                old_Sj, old_Uj = J.shape

                # grow / shrink child‐state axis
                if S != old_Sj:
                    if S > old_Sj:
                        pad_rows = torch.zeros((S - old_Sj, old_Uj),
                                               device=J.device)
                        J = torch.cat([J, pad_rows], dim=0)
                    else:
                        pass

                # grow / shrink path axis
                if U != old_Uj:
                    if U > old_Uj:
                        pad_cols = torch.zeros((J.shape[0], U - old_Uj),
                                               device=J.device)
                        J = torch.cat([J, pad_cols], dim=1)
                    else:
                        pass


                self.total_joint_su[l] = J

        for layer in range(self.total_layers):
            self.states[layer] = self.inflate_state(self.states[layer],self.A[layer].shape[0])
            self.current_policy[layer] = self.inflate_policy(self.current_policy[layer],self.B[layer].shape[-1])
            if self.previous_policy is not None:
                self.previous_state[layer] = self.inflate_state(self.previous_state[layer],self.A[layer].shape[0])
                self.previous_policy[layer] = self.inflate_policy(self.previous_policy[layer],self.B[layer].shape[-1])

    def _init_priors(self):
        spatial_shapes = [pdp.num_clusters for pdp in self.spatial_PDPs]
        temporal_shapes = [pdp.num_clusters for pdp in self.temporal_PDPs]
        shapes = spatial_shapes + temporal_shapes

        alpha_gate0 = self.dirichlet_init_val
        device = 'cpu'

        self.A = []
        for S in shapes:
            # shape = (O, S)
            init = alpha_gate0 + torch.rand(S, S, device=device) *.001

            self.A.append(nn.Parameter(init.detach().cpu(), requires_grad=False).detach().cpu())


        self.B = []
        for S, U in zip(shapes, self.num_paths):
            init = alpha_gate0 + torch.rand(S, S, U, device=device)*.001

            self.B.append(nn.Parameter(init.detach().cpu(), requires_grad=False).detach().cpu())


        self.C = []
        for S in shapes:
            init = torch.ones(S,1,device=device)

            self.C.append(nn.Parameter(init.detach().cpu(), requires_grad=False).detach().cpu())


        self.U = []
        for U in self.num_paths:
            init = alpha_gate0 + torch.rand(U, U, device=device)*.001

            self.U.append(nn.Parameter(init.detach().cpu(), requires_grad=False).detach().cpu())


        self.D = []
        for i, S in enumerate(shapes):
            if i < len(shapes) - 1:
                Sp = shapes[i + 1]
                init = alpha_gate0 + torch.rand(S, Sp, device=device)*.001
            else:
                # top–level has no real parent, keep one‐column
                init = alpha_gate0 + torch.rand(S, 1, device=device)*.001

            self.D.append(nn.Parameter(init.detach().cpu(), requires_grad=False).detach().cpu())


        self.E = []
        for i, U in enumerate(self.num_paths):
            if i < len(shapes) - 1:
                Sp = shapes[i + 1]
                init = alpha_gate0 + torch.rand(U, Sp, device=device)*.001

            else:
                init = alpha_gate0 + torch.rand(U, 1, device=device)*.001

            self.E.append(nn.Parameter(init.detach().cpu(), requires_grad=False).detach().cpu())
        self.total_u = [nn.Parameter(torch.zeros(path,1),requires_grad=False) for path in self.num_paths]
        self.total_joint_su = [nn.Parameter(torch.zeros(self.num_paths[i+1],shapes[i]),requires_grad=False) for i in range(len(shapes)-1)]


    def _propose_path_split(self): #currently not used
        num_non_motor_temporal_layers = len(self.temporal_PDPs)-1
        temporal_B, temporal_U, temporal_E      = self.B[-num_non_motor_temporal_layers:], self.U[-num_non_motor_temporal_layers:], self.E[-num_non_motor_temporal_layers:]
        total_u,total_joint_su = self.total_u[-num_non_motor_temporal_layers:],self.total_joint_su[-num_non_motor_temporal_layers]

        for i, (B,U,E,usage,joint_su) in enumerate(zip(temporal_B,temporal_U,temporal_E,total_u,total_joint_su)):
            S, paths         = joint_su.shape
            a_state      = torch.ones(S)*self.dirichlet_init_val        # prior for B/E columns  (feel free to keep a tensor)
            a_path       = torch.ones(paths)*self.dirichlet_init_val        # prior for C rows/cols

            # loop over all current paths
            for u in range(paths):
                n_state = joint_su[:, u]                    # S counts
                if n_state.sum() < 1:                       # never used
                    continue


                n1, n2 = self.propose_partition(n_state)         # each shape (S,)

                dF_state = self.deltaF_dirichlet(n_state, n1, n2, a_state)


                nU_col   = U[:, u].clone() * 0.0
                dF_path  = self.deltaF_dirichlet(nU_col, nU_col, nU_col, a_path)

                # 4) keep split only if global deltaF < 0
                if dF_state + dF_path < 0:
                    B = self.add_dirichlet_column(B, u)            # adds new slice
                    E = self.add_dirichlet_column(E, u)            # adds new row
                    U = self.add_dirichlet_rowcol(U, u)            # adds row & col
                    joint_su = torch.cat([joint_su,
                                           n2.unsqueeze(1)], dim=1)
                    usage    = torch.cat([usage,
                                           torch.tensor([0.])], dim=0)



    def deltaF_dirichlet(self,n, n1, n2, a):
        # n , n1 , n2  shape (...,K)
        lg  = torch.lgamma
        A   = a.sum();  A1 = A + n1.sum();  A2 = A + n2.sum()
        term = lambda nn,AA: lg(AA) - lg(a+nn).sum()
        return term(torch.zeros_like(n),A) \
             - term(n1, A1) - term(n2, A2)

    def propose_partition(self,hist: torch.Tensor):

        if hist.ndim != 1:
            raise ValueError("hist must be a 1-D tensor")

        S = hist.size(0)
        total = hist.sum()
        if total < 2:
            return hist.clone(), torch.zeros_like(hist)


        order = hist.argsort(descending=True)
        cumsum = hist[order].cumsum(0)


        left_mask = cumsum <= total // 2

        if left_mask.all():
            left_mask[-1] = False

        n1 = torch.zeros_like(hist)
        n2 = torch.zeros_like(hist)
        n1[order[left_mask]] = hist[order[left_mask]]
        n2[order[~left_mask]] = hist[order[~left_mask]]

        return n1, n2

    def add_dirichlet_column(self,tensor: torch.Tensor, index: int, axis: int = -1) -> torch.Tensor:

        # Extract the slice to clone
        col = tensor.index_select(dim=axis, index=torch.tensor([index], device=tensor.device))
        # Concatenate along the specified axis
        return torch.cat([tensor, col], dim=axis)

    def add_dirichlet_rowcol(self,tensor: torch.Tensor, index: int) -> torch.Tensor:

        if tensor.ndim != 2:
            raise ValueError("clone_dirichlet_rowcol requires a 2D tensor")

        # 1) clone column at `index`
        col = tensor[:, index].clone()  # shape (U, 1)
        tensor = torch.cat([tensor, col], dim=1)  # shape (U, U+1)

        # 2) clone row at `index`
        row = tensor[index, :].clone()  # shape (1, U+1)
        tensor = torch.cat([tensor, row], dim=0)  # shape (U+1, U+1)

        return tensor

    def update_priors(self,observations, alpha_gate=512, eta=1,reward=None):
        num_blocks = [64,16,4,1,1,1]
        # -- A & D ---------------------------------------------------------
        states_upper = self.states[1:].copy() + [None]
        states_lower = [None] + self.states[1:].copy()
        reward = reward.detach().cpu() if reward is not None else None

        for l, (Al, Dl, obs_prob, q_S,q_S_upper, q_S_lower) in enumerate(zip(self.A,self.D,observations,self.states,states_upper,states_lower)):

            num_children = q_S_lower.shape[0]//q_S_upper.shape[0] if q_S_upper is not None and q_S_lower is not None else None
            if obs_prob is not None:  # A : O × S
                deltaA = obs_prob.T.to('cpu') @ q_S
                Al += eta/num_blocks[l] * self.gate_A(Al, deltaA,l, alpha_gate) * deltaA

            if q_S_lower is not None:  # D : S_lower × S_upper
                deltaD = torch.einsum('bpc,bs->cs', q_S_lower.view(-1,num_children,q_S_lower.shape[-1]), q_S_upper) if q_S_upper is not None else q_S_lower.T.sum(dim=-1,keepdim=True)

                Dl += eta/num_blocks[l] * self.gate(Dl, deltaD, alpha_gate) * deltaD

        # -- B -------------------------------------------------------------
        for l, (Bl, q_S_prev, q_S_next, q_U_now) in enumerate(zip(self.B,self.previous_state,self.states,self.previous_policy)):
            if q_S_prev is None:
                continue
            deltaB = torch.einsum('bs,bt,bu->stu', q_S_prev, q_S_next, q_U_now)
            Bl += eta/num_blocks[l] * self.gate(Bl, deltaB, alpha_gate) * deltaB  # B : S_now × S_next × U_now

        # -- C -------------------------------------------------------------
        for l, (Cl, q_U_now, q_U_next) in enumerate(zip(self.U,self.previous_policy,self.current_policy)):
            if q_U_now is None:
                continue
            deltaC = q_U_now.T @ q_U_next  # C : U_now × U_next
            Cl += eta/num_blocks[l] * self.gate(Cl, deltaC, alpha_gate) * deltaC

        # -- E -------------------------------------------------------------
        for l, (El, q_U_lower, q_S_upper) in enumerate(zip(self.E,self.current_policy,states_upper)):


            num_children = q_U_lower.shape[0]//q_S_upper.shape[0] if q_S_upper is not None else None

            deltaE = torch.einsum('bpc,bs->cs', q_U_lower.view(-1,num_children,q_U_lower.shape[-1]), q_S_upper) if q_S_upper is not None else q_U_lower.T.sum(dim=1,keepdim=True)

            El += eta/num_blocks[l] * self.gate(El, deltaE, alpha_gate) * deltaE

        for l, (Cl, obs)  in enumerate(zip(self.C,observations)):
            if obs is not None and reward is not None:

                obs = obs.view(self.batch_size,-1,self.A[l].shape[0])
                reward = torch.tensor(reward)
                weighted_obs = obs * reward.view(-1,1,1)

                self.C += 1/num_blocks[l]*weighted_obs.view(-1,weighted_obs.shape[-1])

    def gate_A(self, A, deltaA,idx, alpha_gate: float = 512.0, eps: float = 1e-10):


        log_pref = self.digamma(dist.Dirichlet(self.C[idx]))


        def FE(a):
            col_sum = a.sum(0, keepdim=True)  #   (1,S)
            p_os = a / col_sum

            MI = (p_os * (torch.digamma(a) - torch.digamma(col_sum))).sum()

            p_o = p_os.sum(1)  # (O,)
            risk = -(p_o * log_pref).sum()

            return MI + risk


        G_no = FE(A)
        G_yes = FE(A + deltaA)

        p_up = torch.softmax(-alpha_gate* torch.stack([G_no, G_yes]), dim=0)[1]

        return p_up
    def gate(self,prior, delta, alpha_gate=512.0):
        """Return scalar p_up in [0,1] that multiplies delta."""

        # mutual‑information term of a Dirichlet •  (vectorised over columns)
        def MI(d):
            col_sum = d.sum(0, keepdim=True)
            return (d * (torch.digamma(d) - torch.digamma(col_sum))).sum()

        G_no = MI(prior)
        G_yes = MI(prior + delta)
        return torch.softmax(-alpha_gate * torch.stack([G_no, G_yes]), 0)[1]  # scalar

    @contextmanager
    def saved_beliefs(self):

        stash = dict(
            states=[t.detach().detach().cpu().clone() for t in self.states],
            prev_states=[t.detach().detach().cpu().clone() if t is not None else None for t in self.previous_state],
            policy=[t.detach().cpu().detach().clone() for t in self.current_policy],
            prev_policy=[t.detach().detach().cpu().clone() if t is not None else None for t in self.previous_policy],
            free_energy= self.free_energy.copy()
        )
        try:
            yield
        finally:
            self.states = stash["states"]
            self.previous_state = stash["prev_states"]
            self.current_policy = stash["policy"]
            self.previous_policy = stash["prev_policy"]
            self.free_energy = stash["free_energy"]

    def rollout_predictive_trajectory(self, horizon=1, gamma=32.0):

        device = self.device
        L = self.total_layers


        self.pred_state = [[] for _ in range(L)]
        self.pred_policy = [[] for _ in range(L)]

        for l in range(L):
            self.pred_state[l].append(self.states[l].detach().cpu())
            self.pred_policy[l].append(self.current_policy[l].detach().cpu())


        for t in range(horizon):
            ticks = [False] * L


            for l in self.layer_order:

                ticks[l] = True

                if t != 0 and l !=0:
                    self._get_messages(
                        obs=None,
                        layer_idx=l,
                        tick=ticks,
                        layer_type=self.layer_types[l],
                        plan_gamma=gamma
                    )

                self.previous_state[l] = self.states[l].detach().cpu()
                self.previous_policy[l] = self.current_policy[l].detach().cpu()


                Bl = self.B[l].to(device)  # transition tensor
                Ul = self.U[l].to(device)  # policy‐transition


                weighted_T = self.normalize_dist(
                    torch.einsum('snu,bu->bns', Bl, self.current_policy[l].to(device))
                )
                self.states[l] = (
                    torch.einsum('bns,bs->bn', weighted_T, self.states[l].to(device))
                ).detach().cpu()

                # next‐step policy
                self.current_policy[l] = (
                        self.current_policy[l].to(device)
                        @ self.normalize_dist(Ul)
                ).detach().cpu()


                self.pred_state[l].append(self.states[l])
                self.pred_policy[l].append(self.current_policy[l])


                ticks[l] = False


