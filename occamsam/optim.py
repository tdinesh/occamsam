import cvxpy as cp
import cvxpy.atoms
from cvxpy.atoms import norm, mixed_norm, sum_squares
from cvxpy.atoms.affine.vec import vec

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

import equivalence
from factorgraph import GaussianFactorGraph


class WeightedLeastSquares(object):

    def __init__(self, graph):

        assert isinstance(graph, GaussianFactorGraph), "Expected type GaussainFactorGraph for graph, got %s" % type(graph)
        self.graph = graph

    def optimize(self):

        Am, Ap, d, sigma_d = self.graph.observation_system()
        Bp, t, sigma_t = self.graph.odometry_system()


class LeastSquares(object):

    def __init__(self, graph):

        assert isinstance(graph, GaussianFactorGraph), "Expected type GaussainFactorGraph for graph, got %s" % type(graph)
        self.graph = graph

        self.M = None
        self.P = None
        self.res_d = None
        self.res_t = None

    def optimize(self):

        points = self.graph.points
        landmarks = self.graph.landmarks

        num_points = len(points)
        point_dim = self.graph.point_dim
        num_landmarks = len(landmarks)
        landmark_dim = self.graph.landmark_dim

        Am, Ap, d, _ = self.graph.observation_system()
        Bp, t, _ = self.graph.odometry_system()

        M = cp.Variable((landmark_dim, num_landmarks))
        P = cp.Variable((point_dim, num_points))
        objective = cp.Minimize(sum_squares(Am * vec(M) + Ap * vec(P) - d) + sum_squares(Bp * vec(P) - t))
        problem = cp.Problem(objective)
        problem.solve()

        self.M = M.value
        self.P = P.value

        m = self.M.ravel(order='F')
        p = self.P.ravel(order='F')

        self.res_d = Am.dot(m) + Ap.dot(p) - d
        self.res_t = Bp.dot(p) - t

    def update(self):

        points = self.graph.points
        landmarks = self.graph.landmarks

        for i, m in enumerate(landmarks):
            if m.position is None:
                m.position = self.M[:, i].copy()
            else:
                m.position[:] = self.M[:, i].copy()

        for i, p in enumerate(points):
            if p.position is None:
                p.position = self.P[:, i].copy()
            else:
                p.position[:] = self.P[:, i].copy()


class Occam(object):

    def __init__(self, graph):

        assert isinstance(graph, GaussianFactorGraph), "Expected type GaussainFactorGraph for graph, got %s" % type(graph)
        self.graph = graph

    def optimize(self):

        points = self.graph.points
        landmarks = self.graph.landmarks

        num_points = len(points)
        point_dim = self.graph.point_dim
        num_landmarks = len(landmarks)
        landmark_dim = self.graph.landmark_dim

        E = equivalence.equivalence_matrix(landmarks)
        W = equivalence.ComposeWeight([equivalence.ExpDistanceWeight(landmarks), equivalence.SumMassWeight(landmarks)]).W
        Am, Ap, d = self.graph.observation_system()
        Bp, t = self.graph.odometry_system()

        M = cp.Variable((num_landmarks, landmark_dim))
        P = cp.Variable((num_points, point_dim))
        objective = cp.Minimize(mixed_norm(W * E * M.T))
        constraints = [norm(Am * vec(M) + Ap * vec(P) - d)]



