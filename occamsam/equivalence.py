import numpy as np
import scipy as sp
import scipy.sparse
from itertools import combinations


class ComposeWeight(object):

    def __init__(self, transforms):
        self.W = transforms[0].W
        for t in transforms[1:]:
            self.W = t.W.dot(self.W)

    def __call__(self, x):
        return self.W.dot(x)


class Identity(object):

    def __init__(self, landmarks):
        n = sum([1 for mi, mj in combinations(landmarks, 2) if mi.class_label == mj.class_label and
                 mi.position is not None and mj.position is not None])
        self.W = sp.sparse.identity(n, format='dia')

    def __call__(self, x):
        return self.W.dot(x)


class ExpDistanceWeight(object):

    def __init__(self, landmarks, sigma=1):
        weights = []
        for mi, mj in combinations(landmarks, 2):
            if mi.class_label != mj.class_label:
                continue
            if mi.position is None or mj.position is None:
                continue
            weights.append(np.exp(-(np.linalg.norm(mi.position - mj.position) / (2 * sigma)) ** 2))
        self.W = sp.sparse.diags(weights)

    def __call__(self, x):
        return self.W.dot(x)


class SumMassWeight(object):

    def __init__(self, landmarks):
        weights = []
        for mi, mj in combinations(landmarks, 2):
            if mi.class_label != mj.class_label:
                continue
            if mi.position is None or mj.position is None:
                continue
            weights.append(mi.mass + mj.mass)
        self.W = sp.sparse.diags(weights)

    def __call__(self, x):
        return self.W.dot(x)


def equivalence_matrix(landmarks):

    landmark_pairs = []
    index_pairs = []
    for (i, mi), (j, mj) in combinations(enumerate(landmarks), 2):
        if mi.class_label != mj.class_label:
            continue
        if mi.position is None or mj.position is None:
            continue
        landmark_pairs.append((mi, mj))
        index_pairs.append((i, j))

    E = sp.sparse.lil_matrix((len(index_pairs), len(landmarks)))
    for row, (i, j) in enumerate(index_pairs):
        E[row, i] = 1
        E[row, j] = -1

    return E.tocsr()
