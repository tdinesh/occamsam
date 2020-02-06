import numpy as np
import scipy as sp
import scipy.sparse
from itertools import combinations


def identity(mi, mj):
    return 1


def exp_distance(mi, mj):
    sigma = 1
    return np.exp(-(np.linalg.norm(mi.position - mj.position) ** 2 / (2 * (sigma ** 2)))) \
        if np.linalg.norm(mi.position - mj.position) < 3 * sigma else 0


def sum_mass(mi, mj):
    return mi.mass + mj.mass


def equivalence_matrix(landmarks, transforms=[identity]):
    """
    Supposing each LandmarkVariable in landmarks corresponds to a node in a graph, this function returns the set of edges
        connecting pairs of plausibly equivalent LandmarkVariables in the form of a transposed incidence matrix, i.e.,
        each row contains exactly two entries, 1 and -1, in the columns corresponding to suspected equivalent landmarks

    A LandmarkVariable's column index in the incidence matrix corresponds to its index within the landmarks list

    Provide transform functions (which accept a pair of landmarks as input and returns a weight as a result) in order
        to weight the different equivalences based on size, appearance, pairwise distance, etc. Examples can be seen
        above

    Weight transforms are automatically composed via multiplication

    Weights that approach zero have their corresponding rows in the incidence matrix removed

    The exp_distance function is the most useful to compose with other transforms as it quickly washes out rows relating
        landmarks that are obviously too far apart

    Providing no other transforms besides identity results in all possible pairs being considered

    :param landmarks: List of LandmarkVariables
    :param transforms: List of function classes for weighting suspected equivalent landmark pairs
    :return E: Incidence matrix of plausible equivalences
    :return W: Diagonal weight matrix
    """

    if len(landmarks) == 0:
        return sp.sparse.csr_matrix((0, 0)), sp.sparse.csr_matrix((0, 0))

    index_pairs = []
    weights = []
    for (i, mi), (j, mj) in combinations(enumerate(landmarks), 2):
        if mi.class_label != mj.class_label:
            continue
        if mi.position is None or mj.position is None:
            continue
        index_pairs.append((i, j))
        weights.append(np.prod([t(mi, mj) for t in transforms]))

    if len(index_pairs) == 0:
        return sp.sparse.csr_matrix((0, 0)), sp.sparse.csr_matrix((0, 0))

    index_pairs = np.array(index_pairs)
    weights = np.array(weights)

    zero_mask = np.logical_not(np.isclose(weights, 0))
    index_pairs = index_pairs[zero_mask, :]
    weights = weights[zero_mask]

    E = sp.sparse.lil_matrix((len(index_pairs), len(landmarks)))
    for row, (i, j) in enumerate(index_pairs):
        E[row, i] = 1
        E[row, j] = -1

    W = sp.sparse.diags(weights)

    return E.tocsr(), W
