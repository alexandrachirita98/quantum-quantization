import torch
from torch.nn import functional as F


def to_bag_of_words(images, vocab_size=784):
    # pixel intensity = word count; like the repo's documents, images are NOT length-normalized
    return images.reshape(-1, vocab_size).float() / 255.


def standard_normal_kld(mu, log_sigma):
    return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)

def covariance_penalty(topic_embeddings):
    norm_topic = F.normalize(topic_embeddings, dim=-1)
    cosine = (norm_topic @ norm_topic.t()).abs()
    mean = cosine.mean()
    var = ((cosine - mean) ** 2).mean()
    return mean + var

def elbo_terms(batch, recon_batch, mu, log_sigma, topic_embeddings):
    recon_loss = -torch.sum(torch.log(torch.clamp(recon_batch, min=1e-16)) * batch, dim=-1)
    kld = standard_normal_kld(mu, log_sigma)
    cov = covariance_penalty(topic_embeddings)
    return recon_loss.mean(), kld.mean(), cov  # loss = sum of the three


def dist_to_image(p):
    return p.reshape(-1, 28, 28)
