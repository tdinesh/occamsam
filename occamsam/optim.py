import cvxpy as cp
import cvxpy.atoms
from cvxpy.atoms import norm, mixed_norm, sum_squares
from cvxpy.atoms.affine.vec import vec
from cvxpy.atoms.affine.binary_operators import matmul

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

        self.M = None
        self.P = None
        self.res_d = None
        self.res_t = None

    def optimize(self):

        num_points = len(self.graph.points)
        point_dim = self.graph.point_dim
        num_landmarks = len(self.graph.landmarks)
        landmark_dim = self.graph.landmark_dim

        Am, Ap, d, sigma_d = self.graph.observation_system()
        Bp, t, sigma_t = self.graph.odometry_system()
        S_d, S_t = sp.sparse.diags(1 / sigma_d), sp.sparse.diags(1 / sigma_t)

        M = cp.Variable((landmark_dim, num_landmarks))
        P = cp.Variable((point_dim, num_points))
        objective = cp.Minimize(
            sum_squares(S_d * (Am * vec(M) + Ap * vec(P) - d)) + sum_squares(S_t * (Bp * vec(P) - t)))
        problem = cp.Problem(objective)
        problem.solve()

        self.M = M.value
        self.P = P.value

        m = self.M.ravel(order='F')
        p = self.P.ravel(order='F')

        self.res_d = Am.dot(m) + Ap.dot(p) - d
        self.res_t = Bp.dot(p) - t

    def update(self):

        for i, m in enumerate(self.graph.landmarks):
            if m.position is None:
                m.position = self.M[:, i].copy()
            else:
                m.position[:] = self.M[:, i].copy()

        for i, p in enumerate(self.graph.points):
            if p.position is None:
                p.position = self.P[:, i].copy()
            else:
                p.position[:] = self.P[:, i].copy()


class LeastSquares(object):

    def __init__(self, graph):

        assert isinstance(graph, GaussianFactorGraph), "Expected type GaussainFactorGraph for graph, got %s" % type(graph)
        self.graph = graph

        self.M = None
        self.P = None
        self.res_d = None
        self.res_t = None

    def optimize(self):

        num_points = len(self.graph.points)
        point_dim = self.graph.point_dim
        num_landmarks = len(self.graph.landmarks)
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

        for i, m in enumerate(self.graph.landmarks):
            if m.position is None:
                m.position = self.M[:, i].copy()
            else:
                m.position[:] = self.M[:, i].copy()

        for i, p in enumerate(self.graph.points):
            if p.position is None:
                p.position = self.P[:, i].copy()
            else:
                p.position[:] = self.P[:, i].copy()


class Occam(object):

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

        E, W = equivalence.equivalence_matrix(landmarks, transforms=[equivalence.sum_mass, equivalence.exp_distance])
        # E, W = equivalence.equivalence_matrix(landmarks, transforms=[equivalence.exp_distance])
        # E, W = equivalence.equivalence_matrix(landmarks, transforms=[equivalence.sum_mass])
        Am, Ap, d, sigma_d = self.graph.observation_system()
        Bp, t, sigma_t = self.graph.odometry_system()

        M = cp.Variable((landmark_dim, num_landmarks))
        P = cp.Variable((point_dim, num_points))
        objective = cp.Minimize(mixed_norm(matmul(matmul(W, E), M.T)))
        constraints = [norm(matmul(Am, vec(M)) + matmul(Ap, vec(P)) - d) <= 3 * np.linalg.norm(sigma_d),
                       norm(matmul(Bp, vec(P)) - t) <= 3 * np.linalg.norm(sigma_t)]
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=True)

        self.M = M.value
        self.P = P.value
