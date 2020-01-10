import cvxpy as cp
import cvxpy.atoms
from cvxpy.atoms import norm, mixed_norm, sum_squares
from cvxpy.atoms.affine.vec import vec

import numpy as np
import scipy as sp
import scipy.sparse

import equivalence
from factorgraph import GaussianFactorGraph


def _sanitized_noise_array(sigma):
    sigma_ = sigma.copy()
    zero_mask = np.isclose(sigma_, 0)
    if np.any(zero_mask):
        sigma_[zero_mask] = 1
    return sigma_


class WeightedLeastSquares(object):

    def __init__(self, graph, solver=None, verbosity=False):

        assert isinstance(graph, GaussianFactorGraph), "Expected type GaussainFactorGraph for graph, got %s" % type(graph)
        self.graph = graph

        self.M = None
        self.P = None
        self.res_d = None
        self.res_t = None

        self._verbosity = verbosity

        if 'GUROBI' in cp.installed_solvers():
            self._solver = 'GUROBI'
        elif 'MOSEK' in cp.installed_solvers():
            self._solver = 'MOSEK'
        else:
            self._solver = 'ECOS'

        if solver is not None:
            self._solver = solver

    def optimize(self):

        num_points = len(self.graph.free_points)
        point_dim = self.graph.point_dim
        num_landmarks = len(self.graph.landmarks)
        landmark_dim = self.graph.landmark_dim

        Am, Ap, d, sigma_d = self.graph.observation_system()
        Bp, t, sigma_t = self.graph.odometry_system()

        S_d, S_t = sp.sparse.diags(1 / _sanitized_noise_array(sigma_d)), sp.sparse.diags(1 / _sanitized_noise_array(sigma_t))

        M = cp.Variable((landmark_dim, num_landmarks))
        P = cp.Variable((point_dim, num_points))
        objective = cp.Minimize(
            sum_squares(S_d * ((Am * vec(M)) + (Ap * vec(P)) - d)) + sum_squares(S_t * ((Bp * vec(P)) - t)))
        problem = cp.Problem(objective)
        problem.solve(verbose=self._verbosity, solver=self._solver)

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

        for i, p in enumerate(self.graph.free_points):
            if p.position is None:
                p.position = self.P[:, i].copy()
            else:
                p.position[:] = self.P[:, i].copy()


class LeastSquares(object):

    def __init__(self, graph, solver=None, verbosity=False):

        assert isinstance(graph, GaussianFactorGraph), "Expected type GaussainFactorGraph for graph, got %s" % type(graph)
        self.graph = graph

        self.M = None
        self.P = None
        self.res_d = None
        self.res_t = None

        self._verbosity = verbosity

        if 'GUROBI' in cp.installed_solvers():
            self._solver = 'GUROBI'
        elif 'MOSEK' in cp.installed_solvers():
            self._solver = 'MOSEK'
        else:
            self._solver = 'ECOS'

        if solver is not None:
            self._solver = solver

    def optimize(self):

        num_points = len(self.graph.free_points)
        point_dim = self.graph.point_dim
        num_landmarks = len(self.graph.landmarks)
        landmark_dim = self.graph.landmark_dim

        Am, Ap, d, _ = self.graph.observation_system()
        Bp, t, _ = self.graph.odometry_system()

        M = cp.Variable((landmark_dim, num_landmarks))
        P = cp.Variable((point_dim, num_points))
        objective = cp.Minimize(sum_squares(Am * vec(M) + Ap * vec(P) - d) + sum_squares(Bp * vec(P) - t))
        problem = cp.Problem(objective)
        problem.solve(verbose=self._verbosity, solver=self._solver)

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

        for i, p in enumerate(self.graph.free_points):
            if p.position is None:
                p.position = self.P[:, i].copy()
            else:
                p.position[:] = self.P[:, i].copy()


class Occam(object):

    def __init__(self, graph, solver=None, verbosity=False):

        assert isinstance(graph, GaussianFactorGraph), "Expected type GaussainFactorGraph for graph, got %s" % type(graph)
        self.graph = graph

        self.M = None
        self.P = None
        self.equivalence_pairs = []
        self.res_d = None
        self.res_t = None

        self._verbosity = verbosity

        if 'GUROBI' in cp.installed_solvers():
            self._solver = 'GUROBI'
        elif 'MOSEK' in cp.installed_solvers():
            self._solver = 'MOSEK'
        else:
            self._solver = 'ECOS'

        if solver is not None:
            self._solver = solver

        self._pre_optimizer = WeightedLeastSquares(graph, solver=solver, verbosity=verbosity)

    def optimize(self):

        self._pre_optimizer.optimize()
        self._pre_optimizer.update()

        points = self.graph.free_points
        landmarks = self.graph.landmarks

        num_points = len(points)
        point_dim = self.graph.point_dim
        num_landmarks = len(landmarks)
        landmark_dim = self.graph.landmark_dim

        E, W = equivalence.equivalence_matrix(landmarks, transforms=[equivalence.sum_mass, equivalence.exp_distance])
        Am, Ap, d, sigma_d = self.graph.observation_system()
        Bp, t, sigma_t = self.graph.odometry_system()

        S_d, S_t = sp.sparse.diags(1 / _sanitized_noise_array(sigma_d)), sp.sparse.diags(1 / _sanitized_noise_array(sigma_t))

        M = cp.Variable((landmark_dim, num_landmarks))
        P = cp.Variable((point_dim, num_points))
        objective = cp.Minimize(mixed_norm(W * E * M.T))
        constraints = [norm((Am * vec(M)) + (Ap * vec(P)) - d) <= 2 * np.linalg.norm(sigma_d + 1e-6),
                       norm((Bp * vec(P)) - t) <= 2 * np.linalg.norm(sigma_t + 1e-6)]
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=self._verbosity, solver=self._solver)

        E_ = E[np.abs(np.linalg.norm(E * M.value.T, axis=1)) < 0.001, :]
        objective = cp.Minimize(
            sum_squares(S_d * ((Am * vec(M)) + (Ap * vec(P)) - d)) + sum_squares(S_t * ((Bp * vec(P)) - t)))
        constraints = [E_ * M.T == 0] if E_.shape[0] > 0 else []
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=self._verbosity, solver=self._solver, warm_start=True)

        self.M = M.value
        self.P = P.value

        m = self.M.ravel(order='F')
        p = self.P.ravel(order='F')

        self.res_d = Am.dot(m) + Ap.dot(p) - d
        self.res_t = Bp.dot(p) - t

        self.equivalence_pairs = [(landmarks[i], landmarks[j]) for (i, j) in E_.tolil().rows]

    def update(self):

        for i, m in enumerate(self.graph.landmarks):
            if m.position is None:
                m.position = self.M[:, i].copy()
            else:
                m.position[:] = self.M[:, i].copy()

        for i, p in enumerate(self.graph.free_points):
            if p.position is None:
                p.position = self.P[:, i].copy()
            else:
                p.position[:] = self.P[:, i].copy()

        self.graph.merge_landmarks(self.equivalence_pairs)


