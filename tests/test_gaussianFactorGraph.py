from unittest import TestCase

import numpy as np

import factorgraph
import simulator
import utilities

class TestGaussianFactorGraph(TestCase):

    def setUp(self):

        self.max_dim = 3
        self.max_points = 200
        self.max_landmarks = 60

    def reset_sim(self, point_dim=None, landmark_dim=None, seed=None):

        np.random.seed(seed)

        if point_dim is None:
            self.point_dim = np.random.choice(np.arange(1, self.max_dim + 1))
        else:
            self.point_dim = point_dim

        if landmark_dim is None:
            self.landmark_dim = np.random.choice(np.arange(1, self.max_dim + 1))
        else:
            self.landmark_dim = landmark_dim

        self.num_points = np.random.choice(np.arange(1, self.max_points + 1))
        self.num_landmarks = np.random.choice(np.arange(1, self.max_landmarks + 1))

        self.sim = simulator.Simulator(self.point_dim, self.landmark_dim, self.num_points, self.num_landmarks)

class TestAdd(TestGaussianFactorGraph):

    def test_add_factor(self):

        self.reset_sim(10)

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

class TestArrayGen(TestGaussianFactorGraph):

    def test_observation_array1(self):

        self.reset_sim(point_dim=1, landmark_dim=1, seed=111)

        fg = utilities.sim_to_factorgraph(self.sim)

        d_graph = fg.observation_array()
        d_sim = np.concatenate([d for _, _, d in zip(*self.sim.observation_factors())])

        self.assertEqual(len(d_graph), len(d_sim))
        self.assertTrue(np.isclose(d_graph, d_sim).all())

    def test_observation_array2(self):

        self.reset_sim(point_dim=3, landmark_dim=3, seed=112)

        fg = utilities.sim_to_factorgraph(self.sim)

        d_graph = fg.observation_array()
        d_sim = np.concatenate([d for _, _, d in zip(*self.sim.observation_factors())])

        self.assertEqual(len(d_graph), len(d_sim))
        self.assertTrue(np.isclose(d_graph, d_sim).all())

    def test_observation_array3(self):

        self.reset_sim(point_dim=3, landmark_dim=1, seed=113)

        fg = utilities.sim_to_factorgraph(self.sim)

        d_graph = fg.observation_array()
        d_sim = np.concatenate([d for _, _, d in zip(*self.sim.observation_factors())])

        self.assertEqual(len(d_graph), len(d_sim))
        self.assertTrue(np.isclose(d_graph, d_sim).all())

    def test_observation_array4(self):

        self.reset_sim(point_dim=1, landmark_dim=3, seed=114)

        fg = utilities.sim_to_factorgraph(self.sim)

        d_graph = fg.observation_array()
        d_sim = np.concatenate([d for _, _, d in zip(*self.sim.observation_factors())])

        self.assertEqual(len(d_graph), len(d_sim))
        self.assertTrue(np.isclose(d_graph, d_sim).all())

    def test_odometry_array1(self):

        self.reset_sim(point_dim=1, landmark_dim=1, seed=121)

        fg = utilities.sim_to_factorgraph(self.sim)

        t_graph = fg.odometry_array()
        t_sim = np.concatenate([np.dot(R, t) for _, R, t in zip(*self.sim.odometry_factors())])

        self.assertEqual(len(t_graph), len(t_sim))
        self.assertTrue(np.isclose(t_graph, t_sim).all())

    def test_odometry_array2(self):

        self.reset_sim(point_dim=3, landmark_dim=3, seed=122)

        fg = utilities.sim_to_factorgraph(self.sim)

        t_graph = fg.odometry_array()
        t_sim = np.concatenate([np.dot(R, t) for _, R, t in zip(*self.sim.odometry_factors())])

        self.assertEqual(len(t_graph), len(t_sim))
        self.assertTrue(np.isclose(t_graph, t_sim).all())

    def test_odometry_array3(self):

        self.reset_sim(point_dim=3, landmark_dim=1, seed=123)

        fg = utilities.sim_to_factorgraph(self.sim)

        t_graph = fg.odometry_array()
        t_sim = np.concatenate([np.dot(R, t) for _, R, t in zip(*self.sim.odometry_factors())])

        self.assertEqual(len(t_graph), len(t_sim))
        self.assertTrue(np.isclose(t_graph, t_sim).all())

    def test_odometry_array4(self):

        self.reset_sim(point_dim=1, landmark_dim=3, seed=124)

        fg = utilities.sim_to_factorgraph(self.sim)

        t_graph = fg.odometry_array()
        t_sim = np.concatenate([np.dot(R, t) for _, R, t in zip(*self.sim.odometry_factors())])

        self.assertEqual(len(t_graph), len(t_sim))
        self.assertTrue(np.isclose(t_graph, t_sim).all())

class TestSystemGen(TestGaussianFactorGraph):

    def test_observation_system1(self):

        self.reset_sim(point_dim=1, landmark_dim=1, seed=211)

        fg = utilities.sim_to_factorgraph(self.sim)






