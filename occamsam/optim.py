import cvxpy as cp
import cvxpy.atoms
from cvxpy.atoms import norm, mixed_norm
from cvxpy.atoms.affine.vec import vec

import equivalence
from factorgraph import GaussianFactorGraph

class Optim(object):

    def __init__(self, graph):

        assert isinstance(graph, GaussianFactorGraph), "Expected type GaussainFactorGraph for graph, got %s" % type(graph)
        self.graph = graph

    def optimize(self):
        pass

    def update(self):
        pass


class Occam(Optim):

    def __init__(self, graph):
        super(Occam).__init__(graph)

    def optimize(self):

        points = self.graph.points
        landmarks = self.graph.landmarks

        num_landmarks = len(landmarks)
        landmark_dim = self.graph.landmark_dim

        E = equivalence.equivalence_matrix(landmarks)
        W = equivalence.ComposeWeight([equivalence.ExpDistanceWeight(landmarks), equivalence.SumMassWeight(landmarks)]).W
        Am, Ap, d = self.graph.observation_system()
        Bp, t = self.graph.odometry_system()

        M = cp.Variable((num_landmarks, landmark_dim))
        p = cp.Variable(Ap.shape[1])
        objective = cp.Minimize(mixed_norm(W * E * M.T))
        constraints = [norm(Am * vec(M) + Ap * p - d)]



