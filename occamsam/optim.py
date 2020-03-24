import cvxpy as cp
import cvxpy.atoms
from cvxpy.atoms import norm, mixed_norm, sum_squares
from cvxpy.atoms.affine.vec import vec

import numpy as np
import scipy as sp
import scipy.sparse

import equivalence
import utilities
from factorgraph import GaussianFactorGraph

import itertools


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

    def update(self, merge=True):

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

        if merge:
            self.graph.merge_landmarks(self.equivalence_pairs)


class EM(object):

    def __init__(self, graph, assoc_range=1, solver=None, verbosity=False):

        assert isinstance(graph, GaussianFactorGraph), "Expected type GaussainFactorGraph for graph, got %s" % type(graph)
        self.graph = graph

        self.M = None  # estimated landmark positions
        self.P = None  # estimated robot positions
        self.res_d = None  # distance measurement residuals for a given solution
        self.res_t = None  # translation measurement residuals for a given solution

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

        pre_optimizer = WeightedLeastSquares(graph, solver=solver, verbosity=verbosity)
        pre_optimizer.optimize()
        pre_optimizer.update()

        self.M = pre_optimizer.M
        self.P = pre_optimizer.P

        self.iter_counter = 0

    def optimize(self):

        self.iter_counter = 0

        Am, Ap, d, sigma_d = self.graph.observation_system()
        sigma_d = _sanitized_noise_array(sigma_d)

        W = np.Inf
        W_ = -np.Inf
        while np.linalg.norm(W - W_) > 1e-3:

            # print(np.linalg.norm(W - W_))

            W_ = W

            W = self._e_step(Am, Ap, d, sigma_d)
            self._m_step(W, Am, Ap, d, sigma_d)

            self.iter_counter += 1

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

    def _e_step(self, Am, Ap, d, sigma_d):

        m = self.M.ravel(order='F')
        p = self.P.ravel(order='F')

        block_rows = self.graph.landmark_dim
        Am = sp.sparse.bsr_matrix(Am, blocksize=[block_rows, self.graph.landmark_dim])
        Ap = sp.sparse.bsr_matrix(Ap, blocksize=[block_rows, self.graph.point_dim])
        b = -Ap.dot(p) + d

        points = self.graph.free_points
        landmarks = self.graph.landmarks

        W = []
        k = 0
        for t, xt in enumerate(points):

            # measurements at timestep t
            ks_t = []
            while k < len(Ap.indices) and t == Ap.indices[k]:
                ks_t.append(k)
                k = k + 1

            if len(ks_t) == 0:
                continue

            # space of data associations
            landmarks_t = [landmarks[Am.indices[kt]] for kt in ks_t]
            Dt = self._association_list(landmarks, landmarks_t)

            # probability of each data association
            Am_copy = Am.copy()
            p_z_xld = np.zeros(len(Dt))
            for di, dt in enumerate(Dt):
                p_z_xld[di] = self._model_probability(Am_copy, b, sigma_d, m, ks_t, dt)
            p_z_xl = np.sum(p_z_xld)

            # for each measurement
            Wt = np.zeros((len(ks_t), len(landmarks)))
            for ki in range(len(ks_t)):

                # for each landmark
                for j in range(len(landmarks)):

                    # subset of data associations
                    for di, dt in enumerate(Dt):
                        if dt[ki] == j:
                            Wt[ki, j] += p_z_xld[di] / p_z_xl

            W.append(Wt)

        return np.concatenate(W, axis=0)

    def _m_step(self, W, Am, Ap, d, sigma_d):

        num_points = len(self.graph.free_points)
        point_dim = self.graph.point_dim
        num_landmarks = len(self.graph.landmarks)
        landmark_dim = self.graph.landmark_dim

        Bp, t, sigma_t = self.graph.odometry_system()
        S_t = sp.sparse.diags(1 / _sanitized_noise_array(sigma_t))

        sigma_d = np.tile(_sanitized_noise_array(sigma_d), num_landmarks)
        S_d = sp.sparse.diags(1 / sigma_d)
        W = sp.sparse.diags(W.flatten('F'))

        Am = [np.zeros((Am.shape[0], Am.shape[1])) for _ in range(num_landmarks)]
        for j in range(num_landmarks):
            Am[j][:, j] = 1
        Am = sp.sparse.csr_matrix(np.concatenate(Am, axis=0))
        Ap = sp.sparse.vstack([Ap for _ in range(num_landmarks)])

        d = np.tile(d, num_landmarks)

        M = cp.Variable((landmark_dim, num_landmarks))
        P = cp.Variable((point_dim, num_points))
        objective = cp.Minimize(
            sum_squares(W * S_d * ((Am * vec(M)) + (Ap * vec(P)) - d)) + sum_squares(S_t * ((Bp * vec(P)) - t)))
        problem = cp.Problem(objective)
        problem.solve(verbose=self._verbosity, solver=self._solver)

        self.M = M.value
        self.P = P.value

        m = self.M.ravel(order='F')
        p = self.P.ravel(order='F')

        self.res_d = Am.dot(m) + Ap.dot(p) - d
        self.res_t = Bp.dot(p) - t

    @staticmethod
    def _model_probability(Am, b, sigma_d, m, ks_t, dt):

        Am.indices[Am.indptr[ks_t[0]]:Am.indptr[ks_t[-1]+1]] = dt

        r = (Am.dot(m) - b)
        rk = r[ks_t[0]*Am.blocksize[0]:(ks_t[-1]+1)*Am.blocksize[0]]
        sigma_dk = sigma_d[ks_t[0]*Am.blocksize[0]:(ks_t[-1]+1)*Am.blocksize[0]]

        S_dk = np.diag(1 / sigma_dk**2)
        p = np.exp(-0.5 * np.dot(np.dot(rk, S_dk), rk))

        return p


    @staticmethod
    def _association_list(landmarks, landmarks_t):

        num_landmarks = len(landmarks)
        num_measurements = len(landmarks_t)

        D = []
        for Dt in itertools.product(*([range(num_landmarks)]*num_measurements)):
            violation = False
            for k, j in enumerate(Dt):
                if landmarks_t[k].class_label != landmarks[j].class_label:
                    violation = True
                    break

            if not violation:
                D.append(np.array(Dt))

        return D
