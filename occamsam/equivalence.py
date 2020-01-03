import numpy as np
import scipy as sp
import scipy.sparse
from itertools import combinations


def identity(mi, mj):
    return 1


def exp_distance(mi, mj):
    sigma = 1
    return np.exp(-(np.linalg.norm(mi.position - mj.position) / (2 * sigma)) ** 2)


def sum_mass(mi, mj):
    return mi.mass + mj.mass


def equivalence_matrix(landmarks, transforms=[identity]):
    index_pairs = []
    weights = []
    for (i, mi), (j, mj) in combinations(enumerate(landmarks), 2):
        if mi.class_label != mj.class_label:
            continue
        if mi.position is None or mj.position is None:
            continue
        index_pairs.append((i, j))
        weights.append(np.prod([t(mi, mj) for t in transforms]))

    index_pairs = np.array(index_pairs)
    weights = np.array(weights)

    zero_mask = np.isclose(weights, np.zeros(len(weights)))
    index_pairs = index_pairs[zero_mask, :]
    weights = weights[zero_mask]

    E = sp.sparse.lil_matrix((len(index_pairs), len(landmarks)))
    for row, (i, j) in enumerate(index_pairs):
        E[row, i] = 1
        E[row, j] = -1

    W = sp.sparse.diags(weights)

    return E.tocsr(), W
