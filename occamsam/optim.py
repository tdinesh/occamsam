from itertools import combinations

import cvxpy as cp
import numpy as np
import scipy as sp
import scipy.sparse


def equivalence_matrix(landmarks1):
    """
    Returns the linear system of suspected equivalence constraints on the landmark variables
        E * M = 0


    :return: E:
    """

    distance_threshold = 1e-6
    landmarks = landmarks1
    num_landmarks = len(landmarks)
    suspected_equivalences = [(i, j) for i, j in combinations(range(num_landmarks), 2)
                              if landmarks[i].position is not None and landmarks[j].position is not None
                              and np.linalg.norm(landmarks[i].position - landmarks[j].position) < distance_threshold]

    E = sp.sparse.lil_matrix((len(suspected_equivalences), num_landmarks))
    for i, (j1, j2) in enumerate(suspected_equivalences):
        E[i, j1] = 1
        E[i, j2] = -1

    return E.tocsr(), suspected_equivalences