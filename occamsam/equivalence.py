import numpy as np
import scipy as sp
import scipy.sparse
from itertools import combinations


class Compose(object):

    def __init__(self, transforms):
        self.W = transforms[0].W
        for t in transforms[1:]:
            self.W = t.W.dot(self.W)

    def __call__(self, x):
        return self.W.dot(x)


class Identity(object):

    def __init__(self, equiv_pairs):
        n = len(equiv_pairs)
        self.W = sp.sparse.identity(n, format='dia')

    def __call__(self, x):
        return self.W.dot(x)


class ExpDistanceWeight(object):

    def __init__(self, equiv_pairs, sigma=1):
        weights = []
        for mi, mj in equiv_pairs:
            weights.append(np.exp(-(np.linalg.norm(mi.postion - mj.position) / (2 * sigma))**2))
        self.W = sp.sparse.diags([weights], format='dia')

    def __call__(self, x):
        return self.W.dot(x)


class SumMassWeight(object):

    def __init__(self, equiv_pairs):
        weights = []
        for mi, mj in equiv_pairs:
            weights.append(mi.mass + mj.mass)
        self.W = sp.sparse.diags([weights], format='dia')

    def __call__(self, x):
        return self.W.dot(x)


def equivalence_matrix(landmarks, weight_function=None):

    distance_threshold = 1e-6
    num_landmarks = len(landmarks)
    suspected_equivalences = [(i, j) for i, j in combinations(range(num_landmarks), 2)
                              if i != j and landmarks[i].position is not None and landmarks[j].position is not None
                              and landmarks[i].class_label == landmarks[j].class_label
                              and np.linalg.norm(landmarks[i].position - landmarks[j].position) < distance_threshold]

    E = sp.sparse.lil_matrix((len(suspected_equivalences), num_landmarks))
    for i, (j0, j1) in enumerate(suspected_equivalences):
        E[i, j0] = 1
        E[i, j1] = -1

    return E.tocsr(), suspected_equivalences
