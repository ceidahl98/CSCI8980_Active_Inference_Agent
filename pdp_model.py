import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import copy, math, itertools

import matplotlib.pyplot as plt
import numpy as np
import copy
from torch.special import gammaln
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
import matplotlib as mpl
from dataclasses import dataclass

@dataclass
class PDPConfig:
    prior_mu: float = 3.0
    prior_sigma: float = .1
    a: float = .99
    b: float = 1000
    prior_kappa: float = 1e-6
    prior_alpha: float = 2
    prior_beta: float = 1
    num_iters: int = 1
    sep_quantile: float = 0.1
    sep_density: float = 0.1
    ema_decay: float = .999
    num_candidates: int = 4
    max_clusters: int = 100
    num_iterations: int = 1
    merge_threshold: float = 0



def logit(x):
    return torch.log(x) - torch.log(1 - x)

class PoissonDirichlet(nn.Module):
    def __init__(self, prior_mu, prior_sigma,
                 a, b,
                 max_clusters=100, num_iterations=10,
                 num_features=12,
                 merge_threshold=.01,
                 prior_kappa=1.0,       # New hyperparameter for NIG prior
                 prior_alpha=2.0,       # New hyperparameter for NIG prior (must be > 1)
                 prior_beta=1.0,
                 sep_quantile=.2,
                 density_quantile=.1,
                 ema_decay = .8,
                 num_candidates = 3):
        super().__init__()
        self.a = a  # discount parameter (0 <= a < 1)
        self.b = b  # concentration parameter (b > -a)
        self.merge_threshold = merge_threshold
        self.sep_quantile = sep_quantile
        self.density_quantile=density_quantile

        if not torch.is_tensor(prior_mu):
            self.prior_mu = torch.tensor(prior_mu, dtype=torch.float32)
        else:
            self.prior_mu = prior_mu.float()
        if not torch.is_tensor(prior_sigma):
            self.prior_sigma = torch.tensor(prior_sigma, dtype=torch.float32)
        else:
            self.prior_sigma = prior_sigma.float()


        self.prior_kappa = torch.tensor(prior_kappa, dtype=torch.float32)
        self.prior_alpha = torch.tensor(prior_alpha, dtype=torch.float32)
        self.prior_beta = torch.tensor(prior_beta, dtype=torch.float32)

        self.max_clusters = max_clusters
        self.num_iterations = num_iterations
        self.D = num_features
        init_clusters=0
        self.clusters = []
        for i in range(init_clusters):
            initial_mu = self.sample_lambda(self.prior_mu, num_features)
            initial_sigma = self.prior_sigma.clone()
            self.clusters.append([initial_mu, initial_sigma, 1,
                                  self.prior_kappa.clone(),
                                  self.prior_alpha.clone(),
                                  self.prior_beta.clone()])

        self.register_buffer("num_clusters", torch.tensor(1, dtype=torch.int32))
        self.register_buffer("cluster_means", torch.zeros((max_clusters, num_features)))
        self.register_buffer("cluster_variances", torch.zeros((max_clusters, num_features)))
        self.register_buffer("cluster_weights", torch.ones(max_clusters) / max_clusters)
        self.total_count = 0
        self.min = 1e12
        self.last_assigned = torch.zeros(max_clusters)  # Track last assignment time

        # Initialize the best free energy and best cluster configuration.
        self.best_free_energy = None
        self.best_clusters = None
        self.t0 = 100
        self.k= .9
        self.iters = 0
        self.global_step=0
        self.fe_ema = None
        self.fe_decay = ema_decay
        self.num_candidates = num_candidates

    def _update_global_fe(self, fe_value, batch_weight=1.0):

        if isinstance(fe_value, torch.Tensor):
            fe_value = fe_value.item()

        if self.fe_ema is None:
            self.fe_ema = fe_value
        else:
            d = self.fe_decay ** batch_weight  # weight‑aware decay
            self.fe_ema = d * self.fe_ema + (1.0 - d) * fe_value

        self.global_step += 1

    def compute_stick_breaking_weights(self, K):
        """
        Compute stick-breaking weights pₖ for k=1,...,K:
            vₖ = (1-a) / (1-a + b + k*a)
        and then:
            p₁ = v₁
            pₖ = vₖ * ∏_{j=1}^{k-1} (1 - vⱼ) for k >= 2.
        """
        v = []
        for k in range(K):
            v_k = (1 - self.a) / (1 - self.a + self.b + k * self.a)
            v.append(v_k)
        v = torch.tensor(v, dtype=torch.float32)
        p = []
        prod = 1.0
        for k in range(K):
            pk = v[k] * prod
            p.append(pk)
            prod = prod * (1 - v[k])
        p = torch.tensor(p, dtype=torch.float32)
        return p

    def initialize_clusters_from_buffers(self):
        K = int(self.num_clusters.item())
        clusters = []
        if K > 1:
            for k in range(K):
                mu = self.cluster_means[k]
                sigma = torch.sqrt(self.cluster_variances[k])
                count = self.cluster_weights[k].item()
                # For backward compatibility we only store [mu, sigma, count] in self.clusters.
                # (The internal update keeps track of the extra hyperparameters.)
                clusters.append([mu.detach().cpu(), sigma.detach().cpu(), count.detach().cpu()])
            self.clusters = clusters

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)
        self.initialize_clusters_from_buffers()

    def sample_lambda(self, rate, num_features):
        """
        Sample a rate vector from a Gamma distribution.

        Args:
            rate: A scalar or tensor representing the rate (inverse scale) parameter.
            num_features: The number of features (dimensions) to sample.

        Returns:
            A tensor of shape (num_features,) containing samples from the Gamma distribution.
        """
        # Ensure that 'rate' is a float tensor.
        if not torch.is_tensor(rate):
            rate = torch.tensor(rate, dtype=torch.float32)
        else:
            rate = rate.float()

        # For each feature dimension, use a fixed concentration parameter.
        # You may adjust this constant if you have a different prior belief.
        concentration = torch.ones(num_features, dtype=torch.float32) * 1.0  # Default shape parameter of 2.0.

        # Expand 'rate' to match num_features if it is a scalar.
        if rate.numel() == 1:
            rate = rate.expand(num_features)
        else:
            rate = rate.expand(num_features)

        # Create a Gamma distribution with the specified concentration and rate.
        gamma_dist = torch.distributions.Gamma(1, .25)


        # Draw a sample using the reparameterization (rsample) method.
        z = gamma_dist.rsample()


        return z

    def sample_from_clusters(self, cluster_weights):
        """
        Vectorized sampling from clusters using soft assignments.
        Instead of selecting one cluster per index, a weighted sum of samples from all clusters is computed.
        """
        num_blocks, C, W, H, num_clusters = cluster_weights.shape

        means = self.cluster_means.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_clusters, D)
        variances = self.cluster_variances.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_clusters, D)
        std = torch.sqrt(variances + 1e-10)

        means = means.expand(num_blocks, C, W, H, num_clusters, -1)
        std = std.expand(num_blocks, C, W, H, num_clusters, -1)

        dist = torch.distributions.Normal(means, std)
        samples_all = dist.rsample()
        weighted_samples = (cluster_weights.unsqueeze(-1) * samples_all).sum(dim=-2)

        return weighted_samples.flatten(start_dim=2, end_dim=3)

    def compute_log_likelihood(self, X, mu, sigma):
        """
        Compute log likelihood for data X under a Gaussian.
        """
        eps = 1e-10
        norm_const = -0.5 * torch.sum(torch.log(2 * math.pi * (sigma ** 2 + eps)))
        quad_term = -0.5 * torch.sum(((X - mu) ** 2) / (sigma ** 2 + eps), dim=1)
        return norm_const + quad_term

    def fit(self, data):


        self.best_free_energy = float('inf')
        self.best_clusters = [
            [
                x.detach().clone() if isinstance(x, torch.Tensor)
                else copy.copy(x)
                for x in cluster
            ]
            for cluster in self.clusters
        ]
        eps = 1e-10
        num_candidates = self.num_candidates
        sep_quantile = self.sep_quantile
        density_quantile = self.density_quantile

        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        if len(data.shape) == 4:
            data = data.permute(0,2,1,3).flatten(start_dim=2)

        B, C, D = data.shape
        assert D == self.D, "Feature dim mismatch"
        X = data.reshape(-1, D); N = X.shape[0]
        if N == 1:
            aug_factor=3
            noise_std=.001
            X_rep = X.detach().unsqueeze(1).repeat(1, aug_factor, 1).view(-1, X.size(1))
            # 2) add tiny noise
            noise = torch.randn_like(X_rep) * noise_std
            X = torch.cat([X, X_rep + noise], dim=0)
            N=N*aug_factor+N

        def fe_nig(n, kappa, alpha, beta):
            return (
                gammaln(alpha) - gammaln(self.prior_alpha)
                + self.prior_alpha * torch.log(self.prior_beta)
                - alpha * torch.log(beta)
                + 0.5 * torch.log(self.prior_kappa / kappa)
                - (n / 2.0) * math.log(2.0 * math.pi)
            )

        def fe_lik(r_mat, log_probs_mat):
            return torch.sum(r_mat * (torch.log(r_mat + eps) - log_probs_mat))

        for it in range(self.num_iterations):
            K = len(self.clusters)

            # log‑likelihood for existing clusters
            w_curr = self.compute_stick_breaking_weights(K)
            log_w_curr = torch.log(w_curr + eps).to(X.device)
            log_liks_exist = torch.stack([
                self.compute_log_likelihood(X, mu.to(X.device), sig.to(X.device)) for (mu, sig, *_) in self.clusters
            ], dim=1) if K else X.new_zeros((N, 0)).to(X.device)

            best_lp = torch.max(log_liks_exist + log_w_curr.unsqueeze(0), dim=1).values if K else X.new_full((N,), -float('inf')).to(X.device)
            residual = -best_lp

            # density‑filtered candidate pool
            with torch.no_grad():

                dmat = torch.cdist(X, X)
                density = 1.0 / (dmat.topk(k=min(6, N), largest=False).values[:, 1:].mean(1) + eps)
                keep = density > torch.quantile(density, density_quantile)
                res_masked = residual.clone(); res_masked[~keep] = -float('inf')
                ordered = torch.argsort(res_masked, descending=True)


                min_sep = getattr(self, 'min_candidate_separation', None)
                if min_sep is None:
                    d_keep = dmat[keep][:, keep]
                    min_sep = torch.quantile(d_keep[d_keep > 0], sep_quantile).item()
                sel = []
                for idx in ordered.tolist():
                    if len(sel) >= num_candidates:
                        break
                    if not keep[idx]:
                        continue
                    if all(torch.norm(X[idx] - X[j]).item() > min_sep for j in sel):
                        sel.append(idx)
                top_idx = torch.tensor(sel, device=X.device)

            k_top = top_idx.numel()
            if k_top == 0:
                #print(f"Iter {it}: no candidates after separation constraint")
                continue

            cand_loglik = [self.compute_log_likelihood(X, X[i], self.prior_sigma) for i in top_idx]

            best_fe = self.best_free_energy
            best_subset_records = None
            best_log_probs = None
            best_r_all = None

            # enumerate subsets ------------------------------------------
            for r in range(1, k_top + 1):
                for subset in itertools.combinations(range(k_top), r):
                    if K + r > self.max_clusters:
                        continue

                    w_all = self.compute_stick_breaking_weights(K + r)
                    log_w_all = torch.log(w_all + eps).to(X.device)

                    log_probs = torch.cat([
                        log_liks_exist + log_w_all[:K].unsqueeze(0),
                        torch.stack([cand_loglik[j] + log_w_all[K + s].unsqueeze(0) for s, j in enumerate(subset)], dim=1)
                    ], dim=1)

                    r_all = torch.softmax(log_probs, dim=1)
                    tau = .01
                    r_all[r_all < tau] = 0.
                    row_sums = r_all.sum(dim=1, keepdim=True)
                    r_all = r_all / (row_sums + eps)
                    # prior/posterior terms
                    prior_fe = 0.0
                    subset_records = []

                    # existing clusters
                    for k in range(K):
                        r_k = r_all[:, k]
                        n_k = r_k.sum().item()
                        if n_k <= 1e-6:
                            continue
                        mu_bar = (r_k.unsqueeze(1) * X).sum(0) / (n_k + eps)
                        var_bar = (r_k.unsqueeze(1) * (X - mu_bar).pow(2)).sum(0) / (n_k + eps)
                        kappa_k = self.clusters[k][3] + n_k
                        m_k = (self.clusters[k][3] * self.clusters[k][0].to(X.device) + n_k * mu_bar) / (kappa_k + eps)
                        alpha_k = self.clusters[k][4] + n_k / 2.0
                        beta_k = self.clusters[k][5].to(X.device) + 0.5 * (n_k * var_bar) + (
                            self.clusters[k][3] * n_k / (2.0 * (kappa_k + eps))
                        ) * (mu_bar - self.clusters[k][0].to(X.device)).pow(2)
                        prior_fe += torch.sum(fe_nig(n_k, kappa_k, alpha_k, beta_k))

                    # new candidates
                    for s, j in enumerate(subset):
                        r_new = r_all[:, K + s]
                        n_new = r_new.sum().item()
                        if n_new <= 1.0:
                            break
                        mu_bar = (r_new.unsqueeze(1) * X).sum(0) / (n_new + eps)
                        var_bar = (r_new.unsqueeze(1) * (X - mu_bar).pow(2)).sum(0) / (n_new + eps)
                        kappa_c = self.prior_kappa + n_new
                        m_c = (self.prior_kappa * self.prior_mu + n_new * mu_bar) / (kappa_c + eps)
                        alpha_c = self.prior_alpha + n_new / 2.0
                        beta_c = self.prior_beta + 0.5 * (n_new * var_bar) + (
                            self.prior_kappa * n_new / (2.0 * (kappa_c + eps))
                        ) * (mu_bar - self.prior_mu).pow(2)
                        subset_records.append([m_c, torch.sqrt(var_bar + eps), n_new, kappa_c, alpha_c, beta_c])
                        prior_fe += torch.sum(fe_nig(n_new, kappa_c, alpha_c, beta_c))
                    else:
                        # likelihood / entropy term
                        lik_fe = fe_lik(r_all, log_probs)
                        total_fe = prior_fe + lik_fe
                        if total_fe < best_fe:
                            best_fe = total_fe
                            best_subset_records = subset_records
                            best_log_probs = log_probs
                            best_r_all = r_all

            if best_subset_records is not None:
                self.clusters.extend(best_subset_records)
                K = len(self.clusters)
                self.best_free_energy = best_fe.item() if isinstance(best_fe, torch.Tensor) else best_fe
                self.best_clusters = [
                    [
                        x.detach().clone() if isinstance(x, torch.Tensor)
                        else copy.copy(x)
                        for x in cluster
                    ]
                    for cluster in self.clusters
                ]
                #print(f"Iter {it}: accepted {len(best_subset_records)} clusters → FE ↓ to {best_fe:.4f}")
                r_used = best_r_all
                log_probs_used = best_log_probs
            else:
                #print(f"Iter {it}: no subset reduced FE")
                log_probs_used = log_liks_exist + log_w_curr.unsqueeze(0)
                r_used = torch.softmax(log_probs_used, dim=1)


            eff = r_used.sum(0)
            mus = (r_used.T @ X) / (eff.unsqueeze(1) + eps)
            diff = X.unsqueeze(1) - mus.unsqueeze(0)
            vars = (r_used.unsqueeze(2) * diff.pow(2)).sum(0) / (eff.unsqueeze(1) + eps)
            t0 = self.t0
            batch_idx = self.iters
            rho_t = (t0 + batch_idx) ** (-self.k)
            self.iters+=1
            for k in range(K):
                n_k = eff[k].item()
                if n_k <= 1e-6:
                    continue
                mu_old, _, n_old, kappa_old, alpha_old, beta_old = self.clusters[k]
                mu_old = mu_old.to(X.device)
                kappa_old = kappa_old.to(X.device)
                alpha_old = alpha_old.to(X.device)
                beta_old = beta_old.to(X.device)

                kappa_new = kappa_old + n_k
                m_new = (kappa_old * mu_old + n_k * mus[k]) / (kappa_new + eps)
                alpha_new = alpha_old + n_k / 2.0

                beta_new = beta_old + 0.5 * (n_k * vars[k].to(X.device)) + (kappa_old * n_k) / (2.0 * (kappa_new + eps)) * (
                    mus[k].to(X.device) - mu_old).pow(2)
                sigma2 = torch.distributions.InverseGamma(alpha_new.to(X.device), beta_new.to(X.device)).rsample().to(X.device)
                sig_new = torch.sqrt(sigma2 + eps)

                mu_new = torch.distributions.Normal(m_new, torch.sqrt(sigma2 / (kappa_new + eps))).rsample()

                self.clusters[k] = [mu_new.detach().cpu(), sig_new.detach().cpu(), n_old+ n_k, kappa_new, alpha_new, beta_new]

            # full FE after update
            prior_post_fe = 0.0
            for mu, sig, n_k, kappa_k, alpha_k, beta_k in self.clusters:
                prior_post_fe += torch.sum(fe_nig(n_k, kappa_k.to(X.device), alpha_k.to(X.device), beta_k.to(X.device)))
            lik_post_fe = fe_lik(r_used, log_probs_used)
            fe_post = prior_post_fe + lik_post_fe
            self._update_global_fe(fe_post)
            #print(f"Iter {it}: FE after update = {fe_post:.4f}")
            if fe_post < self.best_free_energy:
                # self.best_free_energy = self.fe_ema
                self.best_clusters = [
                    [
                        x.detach().cpu().clone() if isinstance(x, torch.Tensor)
                        else copy.copy(x)
                        for x in cluster
                    ]
                    for cluster in self.clusters
                ]

            else:
                self.clusters = [
                    [
                        x.detach().cpu().clone() if isinstance(x, torch.Tensor)
                        else copy.copy(x)
                        for x in cluster
                    ]
                    for cluster in self.best_clusters
                ]
                #self.clusters = copy.deepcopy(self.best_clusters)


        final_assignments, active_idx, soft_assignments = self.assign_clusters(data)
        self.update_cluster_params()

        return final_assignments, active_idx, eff, soft_assignments


    def update_cluster_params(self):

        K = len(self.clusters)
        means = torch.zeros((self.max_clusters, self.D))
        variances = torch.zeros((self.max_clusters, self.D))
        weights = torch.zeros(self.max_clusters)
        for k, (mu, sigma, count, *_) in enumerate(self.clusters):
            means[k] = mu
            variances[k] = sigma ** 2  # Store variance
            weights[k] = count
        weights = weights / weights.sum()
        self.cluster_means[:K] = means[:K]
        self.cluster_variances[:K] = variances[:K]
        self.cluster_weights[:K] = weights[:K]
        self.num_clusters.data = torch.tensor(K, dtype=torch.int32)

    def merge_clusters(self):

        merged = True
        while merged:
            merged = False
            new_clusters = []
            skip = set()
            for i in range(len(self.clusters)):
                if i in skip:
                    continue
                mu_i, sigma_i, count_i, kappa_i, alpha_i, beta_i = self.clusters[i]
                merged_mu = mu_i.clone()
                merged_sigma2 = sigma_i ** 2
                merged_count = count_i
                merged_kappa = kappa_i.clone()
                merged_alpha = alpha_i.clone()
                merged_beta = beta_i.clone()
                for j in range(i + 1, len(self.clusters)):
                    if j in skip:
                        continue
                    mu_j, sigma_j, count_j, kappa_j, alpha_j, beta_j = self.clusters[j]
                    distance = torch.norm(merged_mu - mu_j).item()
                    if distance < self.merge_threshold:
                        # Merge the sufficient statistics weighted by counts.
                        merged_mu = (merged_mu * merged_count + mu_j * count_j) / (merged_count + count_j)
                        merged_sigma2 = (merged_sigma2 * merged_count + (sigma_j ** 2) * count_j) / (
                                    merged_count + count_j)
                        merged_count += count_j
                        # For simplicity, we choose to average the hyperparameters.
                        merged_kappa = (merged_kappa * merged_count + kappa_j * count_j) / (merged_count + count_j)
                        merged_alpha = (merged_alpha * merged_count + alpha_j * count_j) / (merged_count + count_j)
                        merged_beta = (merged_beta * merged_count + beta_j * count_j) / (merged_count + count_j)
                        skip.add(j)
                        merged = True
                new_sigma = torch.sqrt(merged_sigma2)
                new_clusters.append([merged_mu, new_sigma, merged_count, merged_kappa, merged_alpha, merged_beta])
            self.clusters = new_clusters

    def assign_clusters(self, data):
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        if len(data.shape) == 4:
            data = data.permute(0,2,1,3).flatten(start_dim=2)
        batch_size, channels, D = data.shape
        assert D == self.D, "Data feature dimension must match prior dimension."
        X = data.reshape(-1, D)
        K = len(self.clusters)
        cluster_weights = self.compute_stick_breaking_weights(K)
        log_weights_existing = torch.log(cluster_weights[:K] + 1e-10).to(X.device)
        log_liks = []
        for k, cl in enumerate(self.clusters):
            mu_k, sigma_k, _,_,_,_ = cl
            log_lik_k = self.compute_log_likelihood(X, mu_k.to(X.device), sigma_k.to(X.device))
            log_liks.append(log_lik_k)
        log_liks = torch.stack(log_liks, dim=1).to(X.device)
        log_prob_existing = log_liks + log_weights_existing.unsqueeze(0)
        final_assignments = torch.argmax(log_prob_existing, dim=1)
        soft_assignments = torch.softmax(log_prob_existing,dim=1)
        final_assignments = final_assignments.reshape(batch_size, channels)
        unique_vals, counts = torch.unique(final_assignments, return_counts=True)
        mask = torch.zeros_like(self.last_assigned, dtype=torch.bool)
        mask[unique_vals] = True
        self.last_assigned[mask] = 0
        self.last_assigned[~mask] += 1
        #active_indices = self.remove_inactive_clusters(10)
        active_indices=None
        return final_assignments, active_indices, soft_assignments

    def print_cluster_statistics(self):
        print("\n===== Cluster Statistics =====")
        print(f"Total Clusters: {len(self.clusters)}\n")
        for i, (mu, sigma, count) in enumerate(self.clusters):
            mean_values = ", ".join([f"{v:.4f}" for v in mu.tolist()])
            std_values = ", ".join([f"{v:.4f}" for v in sigma.tolist()])
            print(f"Cluster {i + 1}:")
            print(f"  Mean    : [{mean_values}]")
            print(f"  Std Dev : [{std_values}]")
            print(f"  Count   : {count}\n")

    def remove_inactive_clusters(self, threshold=10):
        active_indices = [i for i in range(len(self.clusters)) if self.last_assigned[i] <= threshold]
        new_K = len(active_indices)
        self.clusters = [self.clusters[i] for i in active_indices]
        self.num_clusters.data = torch.tensor(new_K, dtype=torch.int32)
        self.update_cluster_params()
        return active_indices
# ============================================
# Example usage:
# ============================================

#
# class GammaClusterDataGenerator:
#     """Fixed Gamma‑centred clusters with on‑demand mini‑batch sampling."""
#
#     def __init__(self,
#                  num_clusters: int = 5,
#                  gamma_shape: float = 1.0,
#                  gamma_rate: float = 0.25,
#                  var_range: tuple[float, float] = (.2, 1),
#                  full_cov_prob: float = 0,
#                  seed: int | None = None) -> None:
#         self.rng = np.random.default_rng(seed)
#         self.gamma_scale = 1.0 / gamma_rate
#         self.specs: list[tuple[np.ndarray, np.ndarray]] = []  # (mu, Sigma)
#         for _ in range(num_clusters):
#             mu = self.rng.gamma(shape=gamma_shape, scale=self.gamma_scale, size=2)
#             # covariance
#             var_diag = self.rng.uniform(*var_range, size=2)
#             if self.rng.random() < full_cov_prob:
#                 rho = self.rng.uniform(-0.7, 0.7)
#                 cov = np.array([
#                     [var_diag[0], rho * np.sqrt(var_diag[0] * var_diag[1])],
#                     [rho * np.sqrt(var_diag[0] * var_diag[1]), var_diag[1]]
#                 ])
#             else:
#                 cov = np.diag(var_diag)
#             self.specs.append((mu, cov))
#
#     def sample_batch(self,
#                      n_range: tuple[int, int] = (30, 300),
#                      return_labels: bool = False) -> tuple[torch.Tensor, np.ndarray] | torch.Tensor:
#         data_list, lab_list = [], []
#         for k, (mu, cov) in enumerate(self.specs):
#             n_k = self.rng.integers(*n_range)
#             pts = self.rng.multivariate_normal(mu, cov, size=n_k)
#             data_list.append(pts)
#             lab_list.append(np.full(n_k, k, dtype=int))
#         data_np = np.vstack(data_list)
#         labels = np.concatenate(lab_list)
#         data_t = torch.tensor(data_np, dtype=torch.float32).unsqueeze(1)  # (N,1,2)
#         return (data_t, labels) if return_labels else data_t
#
#     # quick accessors
#     @property
#     def centres(self):
#         return np.vstack([mu for mu, _ in self.specs])
#
#

# #
#
# prior_mu = [4, 4]
# prior_sigma = [.1, .1]
#
# a, b = .999, 1000
# prior_kappa, prior_alpha, prior_beta = 1e-6, 2, 1
# sep_quant,sep_density = .1,.05
# ema_decay=.1
# num_candidates = 4
#
# pdp_model = PoissonDirichlet(prior_mu, prior_sigma, a, b,
#                              max_clusters=100,
#                              num_iterations=1,
#                              num_features=2,
#                              merge_threshold=0.5,
#
#                              prior_kappa=prior_kappa,
#                              prior_alpha=prior_alpha,
#                              prior_beta=prior_beta,
#                              sep_quantile=sep_quant,
#                              density_quantile=sep_density,
#                              ema_decay=ema_decay,
#                              num_candidates=num_candidates)
#

# gen = GammaClusterDataGenerator(num_clusters=9, seed=53)
#
# num_outer_iters = 5
# all_data, all_assign = [], []
#
# for it in range(num_outer_iters):
#     batch, _ = gen.sample_batch(n_range=(50, 100), return_labels=True)
#     final_assign, _, eff_cnts, _ = pdp_model.fit(batch)  # batch shape (N,1,2)
#     all_data.append(batch.squeeze(1))      # (N,2)
#     all_assign.append(torch.tensor(final_assign))
#     print(f"Iter {it}: active clusters {len(pdp_model.clusters)} ; eff_counts min/ max", eff_cnts.min().item(), eff_cnts.max().item())
#

#
# data_plot = torch.cat(all_data, dim=0).numpy()
# assign_plot = torch.cat(all_assign).numpy()
# centres = pdp_model.cluster_means.detach().cpu().numpy()
# variances = pdp_model.cluster_variances.detach().cpu().numpy()
#
# plt.figure(figsize=(8, 6))
# plt.scatter(data_plot[:, 0], data_plot[:, 1], c=assign_plot, cmap="viridis", alpha=0.5)
# plt.scatter(centres[:, 0], centres[:, 1], color="red", marker="x", s=160, label="Cluster means")
# ax = plt.gca()
# for mu, var in zip(centres, variances):
#     width, height = 2*np.sqrt(var)
#     ax.add_patch(Ellipse(mu, width, height, edgecolor='red', fc='none', lw=1.5))
#
# # legend by cluster id
# unique_ids = np.unique(assign_plot)
# cmap = plt.get_cmap('viridis'); norm = mpl.colors.Normalize(vmin=unique_ids.min(), vmax=unique_ids.max())
# patches = [mpatches.Patch(color=cmap(norm(i)), label=f"Cluster {i}") for i in unique_ids]
# plt.legend(handles=patches, title="Assigned clusters", loc='best')
# plt.title("PDP clustering on Gamma‑centred synthetic data (mini‑batch training)")
# plt.show()
