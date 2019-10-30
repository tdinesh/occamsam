from unittest import TestCase

import numpy as np
import networkx as nx
import factorgraph
import matplotlib.pyplot as plt


class TestGaussianFactorGraph(TestCase):

    def setUp(self):
        np.random.seed(10)

        self.fg = factorgraph.GaussianFactorGraph()

        self.pos_dim = 3
        self.lm_dim = 1

        self.num_positions = 100
        self.num_landmarks = 50
        self.num_observations = 1000

        self.positions = 10 * np.random.rand(self.num_positions, self.pos_dim)
        self.landmarks = 10 * np.random.rand(self.num_landmarks, self.lm_dim)

        principal_dirs = np.array([[1., 0., 0.],
                                   [0., 1., 0.],
                                   [0., 0., 1.],
                                   [1., 1., 0.],
                                   [0., 1., 1.]])
        principal_dirs = np.divide(principal_dirs, np.linalg.norm(principal_dirs, axis=1, keepdims=True))
        self.lm_labels = np.random.choice(5, self.num_landmarks)
        self.lm_dirs = principal_dirs[self.lm_labels, :]

        # Enumerate position-landmark pairs for observation factor generation
        obsvids = np.random.choice(self.num_positions * self.num_landmarks, self.num_observations, replace=False)
        pids, lids = np.meshgrid(np.arange(self.num_positions), np.arange(self.num_landmarks))
        pids, lids = np.ravel(pids), np.ravel(lids)

        self.distances = self.landmarks[lids, :] - \
                    np.sum(np.multiply(self.lm_dirs[lids, :], self.positions[pids, :]), axis=1, keepdims=True)



class TestAdd(TestGaussianFactorGraph):

    def test_add_factor(self):
        p1 = factorgraph.PointVariable(3)
        p2 = factorgraph.PointVariable(3)
        p3 = factorgraph.PointVariable(3)

        lm1 = factorgraph.LandmarkVariable(1, 0)
        lm2 = factorgraph.LandmarkVariable(1, 0)
        lm3 = factorgraph.LandmarkVariable(1, 1)

        f1 = factorgraph.OdometryFactor(p1, p2, np.eye(3), np.array([1, 0, 0]))
        f2 = factorgraph.ObservationFactor(p2, lm1, np.array([1, 0, 0]), np.array([-5]))
        f3 = factorgraph.ObservationFactor(p2, lm2, np.array([0, 1, 0]), np.array([10]))
        f4 = factorgraph.OdometryFactor(p2, p3, np.eye(3), np.array([0, 1, 0]))
        f5 = factorgraph.ObservationFactor(p3, lm3, np.array([0, 1, 1]), np.array([12]))

        self.fg.add_factor(f1)
        self.fg.add_factor(f2)
        self.fg.add_factor(f3)
        self.fg.add_factor(f4)
        self.fg.add_factor(f5)

        self.assertEqual(self.fg.graph.number_of_nodes(), 6)
        self.assertEqual(self.fg.graph.number_of_edges(), 5)
