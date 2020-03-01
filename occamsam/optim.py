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
    """
    Replaces zero-noise estimates with 1 to maintain a achieve neutral weight within weighted least-squares

    :param sigma: list of noise estimates for each measurement
    :return sigma_: copy of sigma with 0 entries replaced by 1
    """

    sigma_ = sigma.copy()
    zero_mask = np.isclose(sigma_, 0)
    if np.any(zero_mask):
        sigma_[zero_mask] = 1
    return sigma_


class LeastSquares(object):

    def __init__(self, graph, solver=None, verbosity=False):
        """
        Ordinary Least-Squares optimizer for the odometric and distance measurements contained in a GaussianFactorGraph

        graph instance is modified using the solution found by optimize() with each call to update()

        :param graph: GaussianFactorGraph instance
        :param solver: One of the supported CvxPy solvers, e.g. 'GUROBI' (default1), 'MOSEK' (default2), 'ECOS' (default3)
        :param verbosity: Prints solver output to console if True
        """

        assert isinstance(graph, GaussianFactorGraph), "Expected type GaussainFactorGraph for graph, got %s" % type(graph)
        self.graph = graph

        self.M = []  # estimated landmark positions
        self.P = []  # estimated robot positions
        self.res_d = []  # distance measurement residuals for a given solution
        self.res_t = []  # translation measurement residuals for a given solution

        self._verbosity = verbosity  # solver output printed to console when True

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

        if (num_points != 0) and (num_landmarks != 0):

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

        elif (num_points != 0) and (num_landmarks == 0):

            P = cp.Variable((point_dim, num_points))
            objective = cp.Minimize(sum_squares(Bp * vec(P) - t))

            problem = cp.Problem(objective)
            problem.solve(verbose=self._verbosity, solver=self._solver)

            self.P = P.value

            p = self.P.ravel(order='F')

            self.res_t = Bp.dot(p) - t

        else:
            return


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


class WeightedLeastSquares(object):

    def __init__(self, graph, solver=None, verbosity=False):
        """
        Weighted Least-Squares optimizer for the odometric and distance measurements contained in a GaussianFactorGraph

        Weights for each measurement in the regression are defined as the inverse of the standard deviation for each.
            If and when 0, the corresponding standard deviation is assumed to be 1.

        graph instance is modified using the solution found by optimize() with each call to update()

        :param graph: GaussianFactorGraph instance
        :param solver: One of the supported CvxPy solvers, e.g. 'GUROBI' (default1), 'MOSEK' (default2), 'ECOS' (default3)
        :param verbosity: Prints solver output to console if True
        """

        assert isinstance(graph, GaussianFactorGraph), "Expected type GaussainFactorGraph for graph, got %s" % type(graph)
        self.graph = graph

        self.M = None  # estimated landmark positions
        self.P = None  # estimated robot positions
        self.res_d = None  # distance measurement residuals for a given solution
        self.res_t = None  # translation measurement residuals for a given solution

        self._verbosity = verbosity  # solver output printed to console when True

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

        if (num_points != 0) and (num_landmarks != 0):

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

        elif (num_points != 0) and (num_landmarks == 0):

            P = cp.Variable((point_dim, num_points))
            objective = cp.Minimize(
                sum_squares(sum_squares(S_t * ((Bp * vec(P)) - t))))
            problem = cp.Problem(objective)
            problem.solve(verbose=self._verbosity, solver=self._solver)

            self.P = P.value

            p = self.P.ravel(order='F')

            self.res_t = Bp.dot(p) - t

        else:
            return

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

    def __init__(self, graph, assoc_range=1, solver=None, verbosity=False):
        """
        Occam Smoothing-And-Mapping optimizer for the odometric and distance factors contained in a GaussianFactorGraph

        Corresponding paper explaining the procedure can be found here:

        Landmark associations are uncovered automatically and stored in equivalence_pairs between calls to optimize()

        graph instance is modified using the solution found by optimize() with each call to update()

        :param graph: GaussianFactorGraph instance
        :param assoc_range: Standard deviation (distance) between pairs of observations to the same landmark
        :param solver: One of the supported CvxPy solvers, e.g. 'GUROBI' (default1), 'MOSEK' (default2), 'ECOS' (default3)
        :param verbosity: Prints solver output to console if True
        """

        assert isinstance(graph, GaussianFactorGraph), "Expected type GaussainFactorGraph for graph, got %s" % type(graph)
        self.graph = graph

        self.M = None  # estimated landmark positions
        self.P = None  # estimated robot positions
        self.res_d = None  # distance measurement residuals for a given solution
        self.res_t = None  # translation measurement residuals for a given solution

        self.equivalence_pairs = []  # equivalent LandmarkVariable pairs

        self._sigma = assoc_range

        self._verbosity = verbosity  # solver output printed to console when True

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

        transforms = [equivalence.SumMass(self.graph.correspondence_map.set_map()),
                      equivalence.ExpDistance(self._sigma),
                      equivalence.Facing()]
        E, W = equivalence.equivalence_matrix(landmarks, transforms=transforms)
        if E.shape[0] == 0:
            self.M = self._pre_optimizer.M
            self.P = self._pre_optimizer.P
            self.res_d = self._pre_optimizer.res_d
            self.res_t = self._pre_optimizer.res_t
            self.equivalence_pairs = []
            return

        Am, Ap, d, sigma_d = self.graph.observation_system()
        Bp, t, sigma_t = self.graph.odometry_system()

        S_d, S_t = sp.sparse.diags(1 / _sanitized_noise_array(sigma_d)), sp.sparse.diags(1 / _sanitized_noise_array(sigma_t))

        M = cp.Variable((landmark_dim, num_landmarks))
        P = cp.Variable((point_dim, num_points))

        M.value = self._pre_optimizer.M
        P.value = self._pre_optimizer.P

        objective = cp.Minimize(mixed_norm(W * E * M.T))
        constraints = [norm((Am * vec(M)) + (Ap * vec(P)) - d) <= 2 * np.linalg.norm(sigma_d + 1e-6),
                       norm((Bp * vec(P)) - t) <= 2 * np.linalg.norm(sigma_t + 1e-6)]
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=self._verbosity, solver=self._solver, warm_start=True)

        if problem.solution.status == 'infeasible':
            self.M = self._pre_optimizer.M
            self.P = self._pre_optimizer.P
            self.res_d = self._pre_optimizer.res_d
            self.res_t = self._pre_optimizer.res_t
            self.equivalence_pairs = []
            return

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


