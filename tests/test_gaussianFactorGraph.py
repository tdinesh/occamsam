from unittest import TestCase

import numpy as np
import scipy as sp
import scipy.stats
import networkx as nx

import factorgraph
import simulator

class TestGaussianFactorGraph(TestCase):

    def setUp(self):

        np.random.seed(10)

        self.point_dim = 3
        self.landmark_dim = 1
        self.num_points = 100
        self.num_landmarks = 20

        self.fg = factorgraph.GaussianFactorGraph()
        self.sim = simulator.Simulator(self.point_dim, self.landmark_dim, self.num_points, self.num_landmarks)


class TestAdd(TestGaussianFactorGraph):

    def test_add_factor(self):

        point_variables = [factorgraph.PointVariable(self.sim.point_dim) for _ in range(self.sim.num_points)]
        landmark_variables = [factorgraph.LandmarkVariable(self.sim.landmark_dim, self.sim.landmark_labels[i])
                              for i in range(self.sim.num_landmarks)]

        odometry_factors = [factorgraph.OdometryFactor(point_variables[u], point_variables[v], R, t)
                            for (u, v), R, t in zip(*self.sim.odometry_factors())]
        observation_factors = [factorgraph.ObservationFactor(point_variables[u], landmark_variables[v], H, d)
                               for (u, v), H, d in zip(*self.sim.observation_factors())]

        for factor in odometry_factors:
            self.fg.add_factor(factor)

        for factor in observation_factors:
            self.fg.add_factor(factor)

        self.assertEqual(self.fg.graph.number_of_nodes(), self.num_points + self.num_landmarks)
        self.assertEqual(self.fg.graph.number_of_edges(), len(odometry_factors) + len(observation_factors))

