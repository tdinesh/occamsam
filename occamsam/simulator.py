import numpy as np
import scipy as sp
import scipy.stats
from collections import OrderedDict
from itertools import chain

from variable import PointVariable, LandmarkVariable
from factor import OdometryFactor, ObservationFactor, PriorFactor
from utilities import random_groups, sample_pairs, UnionFind


class Simulation(object):

    def __init__(self, point_dim, landmark_dim, num_points, num_landmarks, observation_noise, odometry_noise, noise_matrix):

        self.point_dim = point_dim
        self.landmark_dim = landmark_dim
        self.num_points = num_points
        self.num_landmarks = num_landmarks
        self.observation_noise = observation_noise
        self.odometry_noise = odometry_noise
        self.noise_matrix = noise_matrix

        self.num_unique_landmarks = int(0.2 * self.num_landmarks)
        self.num_classes = np.random.choice(10) + 1

        self.point_positions = 50 * np.random.rand(self.num_points, self.point_dim)
        point_ids = np.arange(self.num_points).tolist()
        self._point_pairs = list(zip(point_ids, point_ids[1:]))

        self.unique_landmark_positions = 50 * np.random.rand(self.num_unique_landmarks, self.landmark_dim)
        unique_landmark_labels = np.random.choice(self.num_classes, self.num_unique_landmarks)
        if self.point_dim > 1:
            ortho_group = sp.stats.ortho_group
            principal_dirs = [ortho_group.rvs(self.point_dim)[:self.landmark_dim, :] for _ in range(self.num_classes)]
        else:
            rvs = [np.random.rand(self.landmark_dim, 1) for _ in range(self.num_classes)]
            principal_dirs = [np.divide(x, np.linalg.norm(x)) for x in rvs]
        unique_landmark_orientation = np.array([principal_dirs[label] for label in unique_landmark_labels])

        equivalence_groups = random_groups(self.num_landmarks, self.num_unique_landmarks)
        landmark_positions = np.zeros((self.num_landmarks, self.landmark_dim))
        landmark_labels = -np.ones(self.num_landmarks)
        landmark_orientation = np.zeros((self.num_landmarks, self.landmark_dim, self.point_dim))
        unique_index_map = -np.ones(self.num_landmarks, dtype=np.int)
        for i, group in enumerate(equivalence_groups):
            g_list = list(group)
            landmark_positions[g_list, :] = self.unique_landmark_positions[i, :]
            landmark_labels[g_list] = unique_landmark_labels[i]
            landmark_orientation[g_list, :, :] = unique_landmark_orientation[i]
            unique_index_map[g_list] = i
        assert np.all(landmark_labels > -1), "Unset landmark_labels"

        landmark_ids, point_ids = np.meshgrid(np.arange(self.num_landmarks), np.arange(self.num_points))
        landmark_ids, point_ids = np.ravel(landmark_ids), np.ravel(point_ids)
        uncovered_landmarks = set(np.arange(self.num_landmarks))
        all_observation_ids = iter(np.random.permutation(self.num_points * self.num_landmarks))
        observation_pairs = []
        while len(uncovered_landmarks) > 0 or len(observation_pairs) < 2 * self.num_points:
            i = next(all_observation_ids)
            observation_pairs.append((point_ids[i], landmark_ids[i]))
            uncovered_landmarks = uncovered_landmarks.difference({landmark_ids[i]})
        observation_pairs = sorted(observation_pairs, key=lambda x: x[0])

        observation_points, observation_landmarks = list(zip(*observation_pairs))
        landmark_order = list(OrderedDict.fromkeys(observation_landmarks))
        reindexing_map = -np.ones(self.num_landmarks, dtype=np.int)
        reindexing_map[landmark_order] = np.arange(self.num_landmarks)
        assert np.all(reindexing_map > -1), "Unset index in reindexing_map"
        self.landmark_positions = landmark_positions[landmark_order, :]
        self.landmark_labels = landmark_labels[landmark_order]
        self.landmark_orientations = landmark_orientation[landmark_order, :, :]
        self.landmark_masses = 15 * np.random.random(self.num_landmarks)
        self._observation_pairs = list(zip(list(observation_points), reindexing_map[list(observation_landmarks)]))
        self.num_observations = len(self._observation_pairs)

        self._equivalence_groups = [frozenset(reindexing_map[list(group)]) for group in equivalence_groups]
        self.unique_index_map = unique_index_map[landmark_order]

        self.point_variables = [PointVariable(self.point_dim) for _ in range(self.num_points)]
        self.landmark_variables = [LandmarkVariable(self.landmark_dim, self.landmark_labels[i], self.landmark_masses[i])
                                   for i in range(self.num_landmarks)]
        self.fix_points([0])

    def odometry_measurements(self):

        if self.point_dim > 1:
            ortho_group = sp.stats.ortho_group
            rs = [ortho_group.rvs(self.point_dim) for _ in self._point_pairs]
        else:
            rs = [np.array([1.]) for _ in self._point_pairs]

        if self.noise_matrix == 'diag':
            sigma = self.odometry_noise * np.random.rand(len(self._point_pairs), self.point_dim)
        else:
            sigma = self.odometry_noise * np.ones((len(self._point_pairs), self.point_dim))

        ts = []
        for i, (u, v) in enumerate(self._point_pairs):
            t = np.dot(rs[i].T, self.point_positions[v, :] - self.point_positions[u, :]) + \
                np.multiply(sigma[i, :], np.random.randn(self.point_dim))
            if t.size == 1:
                t = t[0]
            ts.append(t)

        return self._point_pairs, rs, ts, sigma

    def observation_measurements(self):

        hs = [self.landmark_orientations[v] for _, v in self._observation_pairs]

        if self.noise_matrix == 'diag':
            sigma = self.observation_noise * np.random.rand(len(self._observation_pairs), self.landmark_dim)
        else:
            sigma = self.observation_noise * np.ones((len(self._observation_pairs), self.landmark_dim))

        ds = []
        for i, (u, v) in enumerate(self._observation_pairs):
            d = self.landmark_positions[v, :] - np.dot(self.landmark_orientations[v], self.point_positions[u, :]) + \
                np.multiply(sigma[i, :], np.random.randn(self.landmark_dim))
            if d.size == 1:
                d = d[0]
            ds.append(d)

        return self._observation_pairs, hs, ds, sigma

    def fix_points(self, point_index):
        for i in point_index:
            self.point_variables[i].position = self.point_positions[i, :]
        return self

    def fix_landmarks(self, landmark_index):
        for i in landmark_index:
            self.landmark_variables[i].position = self.landmark_positions[i, :]
        return self

    def factors(self, point_range=None, max_observations=np.Inf):

        odometry_factors = [OdometryFactor(self.point_variables[u], self.point_variables[v], R, t, sigma)
                            for (u, v), R, t, sigma in zip(*self.odometry_measurements())]
        observation_factors = [ObservationFactor(self.point_variables[u], self.landmark_variables[v], H, d, sigma)
                               for (u, v), H, d, sigma in zip(*self.observation_measurements())]

        i = 0
        j = 0
        init_factor = PriorFactor(self.point_variables[0], np.eye(self.point_dim),
                                  self.point_variables[0].position.copy(), 1e-6 * np.ones(self.point_dim))
        factor_list = []
        for pv in self.point_variables:

            sync_factors = []

            if pv == init_factor.var:
                sync_factors.append(init_factor)

            if pv == odometry_factors[i].head:
                sync_factors.append(odometry_factors[i])
                i += 1

            k = 0
            while j < len(observation_factors) and pv == observation_factors[j].tail and k < max_observations:
                sync_factors.append(observation_factors[j])
                j += 1
                k += 1
            factor_list.append(sync_factors)

        if point_range is not None:
            start, stop = point_range
            factor_list = factor_list[start:stop]

        return chain(*factor_list)

    def equivalences(self, point_range=None):

        if point_range is None:
            start, stop = 0, self.num_points
        else:
            start, stop = point_range

        observed_landmarks = set([landmark_id for point_id, landmark_id in self._observation_pairs if point_id in range(start, stop)])

        equiv_groups = []
        for g in self._equivalence_groups:
            filtered_group = g.intersection(observed_landmarks)
            if filtered_group == frozenset():
                continue
            equiv_groups.append(filtered_group)
        equiv_pairs = sample_pairs(equiv_groups)

        equiv_groups = set(frozenset(self.landmark_variables[i] for i in g) for g in equiv_groups)
        equiv_pairs = [(self.landmark_variables[i] for i in p) for p in equiv_pairs]

        return equiv_groups, equiv_pairs


MAX_DIM = 3
MAX_POINTS = 2000
MAX_LANDMARKS = 500


def new_simulation(point_dim=None, landmark_dim=None, num_points=None, num_landmarks=None, seed=None,
                   observation_noise=0.0, odometry_noise=0.0, noise_matrix='identity'):
    np.random.seed(seed)

    if point_dim is None:
        point_dim = np.random.choice(np.arange(1, MAX_DIM + 1))

    if landmark_dim is None:
        landmark_dim = np.random.choice(np.arange(1, MAX_DIM + 1))

    if num_points is None:
        num_points = np.random.choice(np.arange(np.floor_divide(MAX_POINTS, 5), MAX_POINTS + 1))

    if num_landmarks is None:
        num_landmarks = np.random.choice(
            np.arange(np.floor_divide(MAX_LANDMARKS, 5), MAX_LANDMARKS + 1))

    return Simulation(point_dim, landmark_dim, num_points, num_landmarks, observation_noise, odometry_noise, noise_matrix)
