import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree


def coverage(dist_mat, dim):
    closest = dist_mat.argmin(dim)
    percentage = len(np.unique(closest)) / dist_mat.shape[dim]

    return percentage


def minimum_matching_distance(dist_mat, dim):
    min_dist = dist_mat.min(1 - dim)

    return min_dist.mean()


def two_sample_test(dist_AB, dist_AA, dist_BB, k):
    n_A, n_B = dist_AB.shape[0], dist_AB.shape[1]
    n = n_A + n_B

    dist = np.empty([n_A + n_B, n_A + n_B])
    dist[:n_A, :n_A] = dist_AA
    dist[:n_A, n_A:] = dist_AB
    dist[n_A:, :n_A] = dist_AB.transpose()
    dist[n_A:, n_A:] = dist_BB

    n_edges = (n - 1) * k
    valence = np.zeros(n)
    inner_A = 0
    inner_B = 0
    edge_AB = 0

    for i in range(k):
        Tcsr = minimum_spanning_tree(dist)
        tree = Tcsr.toarray()
        indices = np.array(np.nonzero(tree))
        dist[indices[0], indices[1]] = 1e9
        dist[indices[1], indices[0]] = 1e9
        np.add.at(valence, indices[0], 1)
        np.add.at(valence, indices[1], 1)
        edges = (indices < n_A).sum(0)
        inner_A += (edges == 2).sum()
        inner_B += (edges == 0).sum()
        edge_AB += (edges == 1).sum()

    expected_A = n_edges * n_A * (n_A - 1) / (n * (n - 1))
    expected_B = n_edges * n_B * (n_B - 1) / (n * (n - 1))
    c = ((valence**2).sum() / 2 - n_edges)
    variance_A = expected_A * (1 - expected_A) + \
        (2 * c * n_A * (n_A - 1) * (n_A - 2) / (n * (n - 1) * (n - 2))) + \
        ((n_edges * (n_edges - 1) - 2 * c) *
            (n_A * (n_A - 1) * (n_A - 2) * (n_A - 3)) / (n * (n - 1) * (n - 2) * (n - 3)))

    variance_B = expected_B * (1 - expected_B) + \
        (2 * c * n_B * (n_B - 1) * (n_B - 2) / (n * (n - 1) * (n - 2))) + \
        ((n_edges * (n_edges - 1) - 2 * c) *
            (n_B * (n_B - 1) * (n_B - 2) * (n_B - 3)) / (n * (n - 1) * (n - 2) * (n - 3)))

    covariance = (n_edges * (n_edges - 1) - 2 * c) * (n_A * n_B * (n_A - 1) * (n_B - 1)) \
        / (n * (n - 1) * (n - 2) * (n - 3)) - expected_A * expected_B

    cov = np.array([[variance_A, covariance], [covariance, variance_B]])
    cov_inverse = np.linalg.inv(cov)

    diff_vector = np.array([inner_A - expected_A, inner_B - expected_B])

    # print(diff_vector)

    score = diff_vector.dot(cov_inverse.dot(diff_vector))

    return score


def variance_two_sample(dist_AB, dist_AA, dist_BB, k, iters):
    n_A, n_B = dist_AB.shape[0], dist_AB.shape[1]
    min_points = min(n_A, n_B)

    scores = []
    for i in range(iters):
        perm_A = torch.randperm(n_A)[:min_points]
        perm_B = torch.randperm(n_B)[:min_points]
        score = two_sample_test(
            dist_AB[perm_A, :][:, perm_B], dist_AA[perm_A, :][:, perm_A], dist_BB[perm_B, :][:, perm_B], k
        )
        scores.append(score)

    return np.mean(scores), np.std(scores)
