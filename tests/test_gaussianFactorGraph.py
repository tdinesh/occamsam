from unittest import TestCase

import numpy as np

import factorgraph
import simulator
import utilities


class TestGaussianFactorGraph(TestCase):

    def setUp(self):

        self.max_dim = 3
        self.max_points = 2000
        self.max_landmarks = 80

    def new_sim(self, point_dim=None, landmark_dim=None, num_points=None, num_landmarks=None, seed=None):

        np.random.seed(seed)

        if point_dim is None:
            self.point_dim = np.random.choice(np.arange(1, self.max_dim + 1))
        else:
            self.point_dim = point_dim

        if landmark_dim is None:
            self.landmark_dim = np.random.choice(np.arange(1, self.max_dim + 1))
        else:
            self.landmark_dim = landmark_dim

        if num_points is None:
            self.num_points = np.random.choice(np.arange(np.floor_divide(self.max_points, 5), self.max_points + 1))
        else:
            self.num_points = num_points

        if num_landmarks is None:
            self.num_landmarks = np.random.choice(np.arange(np.floor_divide(self.max_landmarks, 5), self.max_landmarks + 1))
        else:
            self.num_landmarks = num_landmarks

        return simulator.Simulator(self.point_dim, self.landmark_dim, self.num_points, self.num_landmarks)


class TestAdd(TestGaussianFactorGraph):

    def test_add_factor(self):

        sim = self.new_sim(seed=1)

        fg = factorgraph.GaussianFactorGraph()

        point_variables = [factorgraph.PointVariable(sim.point_dim) for _ in range(sim.num_points)]
        landmark_variables = [factorgraph.LandmarkVariable(sim.landmark_dim, sim.landmark_labels[i])
                              for i in range(sim.num_landmarks)]

        odometry_factors = [factorgraph.OdometryFactor(point_variables[u], point_variables[v], R, t)
                            for (u, v), R, t in zip(*sim.odometry_factors())]
        observation_factors = [factorgraph.ObservationFactor(point_variables[u], landmark_variables[v], H, d)
                               for (u, v), H, d in zip(*sim.observation_factors())]

        for factor in odometry_factors:
            fg.add_factor(factor)

        for factor in observation_factors:
            fg.add_factor(factor)

        self.assertEqual(fg._graph.number_of_nodes(), self.num_points + self.num_landmarks)
        self.assertEqual(fg._graph.number_of_edges(), len(odometry_factors) + len(observation_factors))


class TestArrayGen(TestGaussianFactorGraph):

    def test_observation_array1(self):
        sim = self.new_sim(point_dim=1, landmark_dim=1, seed=111)
        fg = utilities.sim_to_factorgraph(sim)

        d_graph = fg.observation_array()
        d_sim = np.concatenate([d for _, _, d in zip(*sim.observation_factors())])

        self.assertEqual(len(d_graph), len(d_sim))
        self.assertTrue(np.isclose(d_graph, d_sim).all())

    def test_observation_array2(self):
        sim = self.new_sim(point_dim=3, landmark_dim=3, seed=112)
        fg = utilities.sim_to_factorgraph(sim)

        d_graph = fg.observation_array()
        d_sim = np.concatenate([d for _, _, d in zip(*sim.observation_factors())])

        self.assertEqual(len(d_graph), len(d_sim))
        self.assertTrue(np.isclose(d_graph, d_sim).all())

    def test_observation_array3(self):
        sim = self.new_sim(point_dim=3, landmark_dim=1, seed=113)
        fg = utilities.sim_to_factorgraph(sim)

        d_graph = fg.observation_array()
        d_sim = np.concatenate([d for _, _, d in zip(*sim.observation_factors())])

        self.assertEqual(len(d_graph), len(d_sim))
        self.assertTrue(np.isclose(d_graph, d_sim).all())

    def test_observation_array4(self):
        sim = self.new_sim(point_dim=1, landmark_dim=3, seed=114)
        fg = utilities.sim_to_factorgraph(sim)

        d_graph = fg.observation_array()
        d_sim = np.concatenate([d for _, _, d in zip(*sim.observation_factors())])

        self.assertEqual(len(d_graph), len(d_sim))
        self.assertTrue(np.isclose(d_graph, d_sim).all())

    def test_odometry_array1(self):
        sim = self.new_sim(point_dim=1, landmark_dim=1, seed=121)
        fg = utilities.sim_to_factorgraph(sim)

        t_graph = fg.odometry_array()
        t_sim = np.concatenate([np.dot(R, t) for _, R, t in zip(*sim.odometry_factors())])

        self.assertEqual(len(t_graph), len(t_sim))
        self.assertTrue(np.isclose(t_graph, t_sim).all())

    def test_odometry_array2(self):
        sim = self.new_sim(point_dim=3, landmark_dim=3, seed=122)
        fg = utilities.sim_to_factorgraph(sim)

        t_graph = fg.odometry_array()
        t_sim = np.concatenate([np.dot(R, t) for _, R, t in zip(*sim.odometry_factors())])

        self.assertEqual(len(t_graph), len(t_sim))
        self.assertTrue(np.isclose(t_graph, t_sim).all())

    def test_odometry_array3(self):
        sim = self.new_sim(point_dim=3, landmark_dim=1, seed=123)
        fg = utilities.sim_to_factorgraph(sim)

        t_graph = fg.odometry_array()
        t_sim = np.concatenate([np.dot(R, t) for _, R, t in zip(*sim.odometry_factors())])

        self.assertEqual(len(t_graph), len(t_sim))
        self.assertTrue(np.isclose(t_graph, t_sim).all())

    def test_odometry_array4(self):
        sim = self.new_sim(point_dim=1, landmark_dim=3, seed=124)
        fg = utilities.sim_to_factorgraph(sim)

        t_graph = fg.odometry_array()
        t_sim = np.concatenate([np.dot(R, t) for _, R, t in zip(*sim.odometry_factors())])

        self.assertEqual(len(t_graph), len(t_sim))
        self.assertTrue(np.isclose(t_graph, t_sim).all())


class TestObservationSystem(TestGaussianFactorGraph):

    def test_full1(self):
        sim = self.new_sim(point_dim=1, landmark_dim=1, seed=211)
        fg = utilities.sim_to_factorgraph(sim)

        A, b = fg.observation_system()
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]), np.ravel(sim.points)))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_full2(self):
        sim = self.new_sim(point_dim=3, landmark_dim=1, seed=212)
        fg = utilities.sim_to_factorgraph(sim)

        A, b = fg.observation_system()
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]), np.ravel(sim.points)))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_full3(self):
        sim = self.new_sim(point_dim=1, landmark_dim=3, seed=213)
        fg = utilities.sim_to_factorgraph(sim)

        A, b = fg.observation_system()
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]), np.ravel(sim.points)))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_full4(self):
        sim = self.new_sim(point_dim=3, landmark_dim=3, seed=214)
        fg = utilities.sim_to_factorgraph(sim)

        A, b = fg.observation_system()
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]), np.ravel(sim.points)))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_marginal1(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = self.new_sim(point_dim=1, landmark_dim=1, num_points=num_points, seed=221)
        fg = utilities.sim_to_factorgraph(sim)

        points = [node for node in fg._graph.nodes() if isinstance(node, factorgraph.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system(num_free_points)
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-num_free_points:, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_marginal2(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = self.new_sim(point_dim=1, landmark_dim=3, num_points=num_points, seed=222)
        fg = utilities.sim_to_factorgraph(sim)

        points = [node for node in fg._graph.nodes() if isinstance(node, factorgraph.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system(num_free_points)
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-num_free_points:, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_marginal3(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = self.new_sim(point_dim=3, landmark_dim=1, num_points=num_points, seed=223)
        fg = utilities.sim_to_factorgraph(sim)

        points = [node for node in fg._graph.nodes() if isinstance(node, factorgraph.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system(num_free_points)
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-num_free_points:, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_marginal4(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = self.new_sim(point_dim=3, landmark_dim=3, num_points=num_points, seed=224)
        fg = utilities.sim_to_factorgraph(sim)

        points = [node for node in fg._graph.nodes() if isinstance(node, factorgraph.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system(num_free_points)
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-num_free_points:, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_over_marginal1(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = self.new_sim(point_dim=1, landmark_dim=1, num_points=num_points, seed=231)
        fg = utilities.sim_to_factorgraph(sim)

        points = [node for node in fg._graph.nodes() if isinstance(node, factorgraph.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system(num_free_points)
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_over_marginal2(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = self.new_sim(point_dim=1, landmark_dim=3, num_points=num_points, seed=232)
        fg = utilities.sim_to_factorgraph(sim)

        points = [node for node in fg._graph.nodes() if isinstance(node, factorgraph.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system(num_free_points)
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_over_marginal3(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = self.new_sim(point_dim=3, landmark_dim=1, num_points=num_points, seed=233)
        fg = utilities.sim_to_factorgraph(sim)

        points = [node for node in fg._graph.nodes() if isinstance(node, factorgraph.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system(num_free_points)
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_over_marginal4(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = self.new_sim(point_dim=1, landmark_dim=3, num_points=num_points, seed=234)
        fg = utilities.sim_to_factorgraph(sim)

        points = [node for node in fg._graph.nodes() if isinstance(node, factorgraph.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system(num_free_points)
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_under_marginal1(self):
        num_points = 2000
        num_fixed_points = 100
        num_free_points = 500

        sim = self.new_sim(point_dim=1, landmark_dim=1, num_points=num_points, seed=241)
        fg = utilities.sim_to_factorgraph(sim)

        points = [node for node in fg._graph.nodes() if isinstance(node, factorgraph.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system(num_free_points)
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_under_marginal2(self):
        num_points = 2000
        num_fixed_points = 242
        num_free_points = 869

        sim = self.new_sim(point_dim=1, landmark_dim=3, num_points=num_points, seed=242)
        fg = utilities.sim_to_factorgraph(sim)

        points = [node for node in fg._graph.nodes() if isinstance(node, factorgraph.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system(num_free_points)
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_under_marginal3(self):
        num_points = 2000
        num_fixed_points = 87
        num_free_points = 419

        sim = self.new_sim(point_dim=3, landmark_dim=1, num_points=num_points, seed=243)
        fg = utilities.sim_to_factorgraph(sim)

        points = [node for node in fg._graph.nodes() if isinstance(node, factorgraph.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system(num_free_points)
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_under_marginal4(self):
        num_points = 2000
        num_fixed_points = 111
        num_free_points = 642

        sim = self.new_sim(point_dim=3, landmark_dim=3, num_points=num_points, seed=244)
        fg = utilities.sim_to_factorgraph(sim)

        points = [node for node in fg._graph.nodes() if isinstance(node, factorgraph.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system(num_free_points)
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())
