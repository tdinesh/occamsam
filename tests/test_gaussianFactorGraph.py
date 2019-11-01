from unittest import TestCase

import numpy as np

import factorgraph
import simulator

class TestGaussianFactorGraph(TestCase):

    def setUp(self):

        np.random.seed(10)

        self.point_dim = 3
        self.landmark_dim = 1
        self.num_points = 100
        self.num_landmarks = 20

        self.sim = simulator.Simulator(self.point_dim, self.landmark_dim, self.num_points, self.num_landmarks)

    def test_add_factor(self):

        fg = factorgraph.GaussianFactorGraph()

        point_variables = [factorgraph.PointVariable(self.sim.point_dim) for _ in range(self.sim.num_points)]
        landmark_variables = [factorgraph.LandmarkVariable(self.sim.landmark_dim, self.sim.landmark_labels[i])
                              for i in range(self.sim.num_landmarks)]

        odometry_factors = [factorgraph.OdometryFactor(point_variables[u], point_variables[v], R, t)
                            for (u, v), R, t in zip(*self.sim.odometry_factors())]
        observation_factors = [factorgraph.ObservationFactor(point_variables[u], landmark_variables[v], H, d)
                               for (u, v), H, d in zip(*self.sim.observation_factors())]

        for factor in odometry_factors:
            fg.add_factor(factor)

        for factor in observation_factors:
            fg.add_factor(factor)

        self.assertEqual(fg.graph.number_of_nodes(), self.num_points + self.num_landmarks)
        self.assertEqual(fg.graph.number_of_edges(), len(odometry_factors) + len(observation_factors))

    def test_observation_array(self):

        fg = factorgraph.GaussianFactorGraph()

        point_variables = [factorgraph.PointVariable(self.sim.point_dim) for _ in range(self.sim.num_points)]
        landmark_variables = [factorgraph.LandmarkVariable(self.sim.landmark_dim, self.sim.landmark_labels[i])
                              for i in range(self.sim.num_landmarks)]

        observation_factors = [factorgraph.ObservationFactor(point_variables[u], landmark_variables[v], H, d)
                               for (u, v), H, d in zip(*self.sim.observation_factors())]

        for factor in observation_factors:
            fg.add_factor(factor)

        d_graph = fg.observation_array()
        d_sim = np.concatenate([d for _, _, d in zip(*self.sim.observation_factors())])

        self.assertEqual(len(d_graph), len(d_sim))

        ## Note: Graph doesn't store edges in the order they were presented
        self.assertTrue(np.isclose(np.sort(d_graph), np.sort(d_sim)).all())

    def test_odometry_array(self):

        fg = factorgraph.GaussianFactorGraph()

        point_variables = [factorgraph.PointVariable(self.sim.point_dim) for _ in range(self.sim.num_points)]

        odometry_factors = [factorgraph.OdometryFactor(point_variables[u], point_variables[v], R, t)
                            for (u, v), R, t in zip(*self.sim.odometry_factors())]

        for factor in odometry_factors:
            fg.add_factor(factor)

        t_graph = fg.odometry_array()
        t_sim = np.concatenate([np.dot(R, t) for _, R, t in zip(*self.sim.odometry_factors())])

        self.assertEqual(len(t_graph), len(t_sim))
        self.assertTrue(np.isclose(t_graph, t_sim).all())

