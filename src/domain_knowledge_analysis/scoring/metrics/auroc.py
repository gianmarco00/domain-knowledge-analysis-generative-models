import torch
from sklearn.metrics import roc_auc_score


def compute_auroc(in_distribution_scores, out_distribution_scores):

    labels = torch.cat([
        torch.zeros(len(in_distribution_scores)),
        torch.ones(len(out_distribution_scores)),
    ])

    scores = torch.cat([
        in_distribution_scores,
        out_distribution_scores,
    ])

    return roc_auc_score(
        labels.numpy(),
        scores.numpy(),
    )