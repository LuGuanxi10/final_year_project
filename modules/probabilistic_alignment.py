# coding=utf-8

import torch
from torch import nn
import torch.nn.functional as F


class GaussianParamHead(nn.Module):
    def __init__(self, embed_dim, logsigma_min=-7.0, logsigma_max=7.0):
        super(GaussianParamHead, self).__init__()
        self.mu_proj = nn.Linear(embed_dim, embed_dim)
        self.logsigma_proj = nn.Linear(embed_dim, embed_dim)
        self.logsigma_min = logsigma_min
        self.logsigma_max = logsigma_max

    def forward(self, x):
        mu = F.normalize(self.mu_proj(x), dim=-1)
        log_sigma = self.logsigma_proj(x)
        log_sigma = torch.clamp(log_sigma, min=self.logsigma_min, max=self.logsigma_max)
        return mu, log_sigma


def sample_gaussian(mu, log_sigma, n_samples):
    std = torch.exp(log_sigma)
    eps = torch.randn(
        n_samples, mu.size(0), mu.size(1), device=mu.device, dtype=mu.dtype
    )
    return mu.unsqueeze(0) + eps * std.unsqueeze(0)


def mc_match_probability(z_txt, z_vid, alpha, beta, eps=1e-6):
    txt_sq = (z_txt ** 2).sum(dim=-1, keepdim=True)
    vid_sq = (z_vid ** 2).sum(dim=-1).unsqueeze(1)
    cross = torch.matmul(z_txt, z_vid.transpose(1, 2))
    dist2 = (txt_sq + vid_sq - 2.0 * cross).clamp(min=0.0)

    sample_logits = -alpha * dist2 + beta
    match_prob = torch.sigmoid(sample_logits).mean(dim=0)
    match_prob = match_prob.clamp(min=eps, max=1.0 - eps)
    sim_logits = torch.logit(match_prob, eps=eps)
    return match_prob, sim_logits


def pair_bce_loss(prob_matrix, target_matrix):
    target_matrix = target_matrix.to(prob_matrix.dtype)
    return F.binary_cross_entropy(prob_matrix, target_matrix)


def gaussian_kl_to_std_normal(mu, log_sigma):
    variance = torch.exp(2.0 * log_sigma)
    kl = 0.5 * (mu.pow(2) + variance - 1.0 - 2.0 * log_sigma)
    return kl.sum(dim=-1).mean()


def uniformity_loss(embeddings, temperature=2.0, eps=1e-8):
    if embeddings.size(0) <= 1:
        return embeddings.new_tensor(0.0)
    embeddings = F.normalize(embeddings, dim=-1)
    dist2 = torch.pdist(embeddings, p=2).pow(2)
    return torch.log(torch.exp(-temperature * dist2).mean() + eps)
