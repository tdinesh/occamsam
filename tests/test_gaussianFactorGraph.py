import unittest

import numpy as np

import factorgraph
import factor
import variable

from simulator import new_simulation
from utilities import sim_to_factorgraph


class TestAdd(unittest.TestCase):

    def test_add_factor(self):

        sim = new_simulation(seed=1)
        fg = factorgraph.GaussianFactorGraph()

        point_variables = [variable.PointVariable(sim.point_dim) for _ in range(sim.num_points)]
        landmark_variables = [variable.LandmarkVariable(sim.landmark_dim, sim.landmark_labels[i])
                              for i in range(sim.num_landmarks)]

        odometry_factors = [factor.OdometryFactor(point_variables[u], point_variables[v], R, t)
                            for (u, v), R, t in zip(*sim.odometry_factors())]
        observation_factors = [factor.ObservationFactor(point_variables[u], landmark_variables[v], H, d)
                               for (u, v), H, d in zip(*sim.observation_factors())]

        for f in odometry_factors:
            fg.add_factor(f)

        for f in observation_factors:
            fg.add_factor(f)

        self.assertEqual(len(fg.variables), sim.num_points + sim.num_landmarks)
        self.assertEqual(len(fg.factors), len(odometry_factors) + len(observation_factors))


class TestObservationArray(unittest.TestCase):

    def test_observation_array1(self):
        sim = new_simulation(point_dim=1, landmark_dim=1, seed=111)
        fg = sim_to_factorgraph(sim)

        _, d_sys = fg.observation_system
        d_sim = np.concatenate([d for _, _, d in zip(*sim.observation_factors())])

        self.assertEqual(len(d_sys), len(d_sim))
        self.assertTrue(np.isclose(d_sys, d_sim).all())

    def test_observation_array2(self):
        sim = new_simulation(point_dim=3, landmark_dim=3, seed=112)
        fg = sim_to_factorgraph(sim)

        _, d_sys = fg.observation_system
        d_sim = np.concatenate([d for _, _, d in zip(*sim.observation_factors())])

        self.assertEqual(len(d_sys), len(d_sim))
        self.assertTrue(np.isclose(d_sys, d_sim).all())

    def test_observation_array3(self):
        sim = new_simulation(point_dim=3, landmark_dim=1, seed=113)
        fg = sim_to_factorgraph(sim)

        _, d_sys = fg.observation_system
        d_sim = np.concatenate([d for _, _, d in zip(*sim.observation_factors())])

        self.assertEqual(len(d_sys), len(d_sim))
        self.assertTrue(np.isclose(d_sys, d_sim).all())

    def test_observation_array4(self):
        sim = new_simulation(point_dim=1, landmark_dim=3, seed=114)
        fg = sim_to_factorgraph(sim)

        _, d_sys = fg.observation_system
        d_sim = np.concatenate([d for _, _, d in zip(*sim.observation_factors())])

        self.assertEqual(len(d_sys), len(d_sim))
        self.assertTrue(np.isclose(d_sys, d_sim).all())


class TestOdometryArray(unittest.TestCase):

    def test_odometry_array1(self):
        sim = new_simulation(point_dim=1, landmark_dim=1, seed=121)
        fg = sim_to_factorgraph(sim)

        _, t_sys = fg.odometry_system
        t_sim = np.concatenate([np.dot(R, t) for _, R, t in zip(*sim.odometry_factors())])

        self.assertEqual(len(t_sys), len(t_sim))
        self.assertTrue(np.isclose(t_sys, t_sim).all())

    def test_odometry_array2(self):
        sim = new_simulation(point_dim=3, landmark_dim=3, seed=122)
        fg = sim_to_factorgraph(sim)

        _, t_sys = fg.odometry_system
        t_sim = np.concatenate([np.dot(R, t) for _, R, t in zip(*sim.odometry_factors())])

        self.assertEqual(len(t_sys), len(t_sim))
        self.assertTrue(np.isclose(t_sys, t_sim).all())

    def test_odometry_array3(self):
        sim = new_simulation(point_dim=3, landmark_dim=1, seed=123)
        fg = sim_to_factorgraph(sim)

        _, t_sys = fg.odometry_system
        t_sim = np.concatenate([np.dot(R, t) for _, R, t in zip(*sim.odometry_factors())])

        self.assertEqual(len(t_sys), len(t_sim))
        self.assertTrue(np.isclose(t_sys, t_sim).all())

    def test_odometry_array4(self):
        sim = new_simulation(point_dim=1, landmark_dim=3, seed=124)
        fg = sim_to_factorgraph(sim)

        _, t_sys = fg.odometry_system
        t_sim = np.concatenate([np.dot(R, t) for _, R, t in zip(*sim.odometry_factors())])

        self.assertEqual(len(t_sys), len(t_sim))
        self.assertTrue(np.isclose(t_sys, t_sim).all())


class TestObservationSystem(unittest.TestCase):

    def test_no_marginal1(self):
        sim = new_simulation(point_dim=1, landmark_dim=1, seed=211)
        fg = sim_to_factorgraph(sim)

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]), np.ravel(sim.points)))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_no_marginal2(self):
        sim = new_simulation(point_dim=3, landmark_dim=1, seed=212)
        fg = sim_to_factorgraph(sim)

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]), np.ravel(sim.points)))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_no_marginal3(self):
        sim = new_simulation(point_dim=1, landmark_dim=3, seed=213)
        fg = sim_to_factorgraph(sim)

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]), np.ravel(sim.points)))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_no_marginal4(self):
        sim = new_simulation(point_dim=3, landmark_dim=3, seed=214)
        fg = sim_to_factorgraph(sim)

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]), np.ravel(sim.points)))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_marginal1(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=1, landmark_dim=1, num_points=num_points, seed=221)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-num_free_points:, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_marginal2(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=222)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-num_free_points:, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_marginal3(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=223)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-num_free_points:, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_marginal4(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=num_points, seed=224)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-num_free_points:, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_over_marginal1(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=1, num_points=num_points, seed=231)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_over_marginal2(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=232)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_over_marginal3(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=233)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_over_marginal4(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=234)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_under_marginal1(self):
        num_points = 2000
        num_fixed_points = 100
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=1, num_points=num_points, seed=241)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_under_marginal2(self):
        num_points = 2000
        num_fixed_points = 242
        num_free_points = 869

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=242)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_under_marginal3(self):
        num_points = 2000
        num_fixed_points = 87
        num_free_points = 419

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=243)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_under_marginal4(self):
        num_points = 2000
        num_fixed_points = 111
        num_free_points = 642

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=num_points, seed=244)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.concatenate((np.ravel(sim.landmarks[sim.observed_landmarks, :]),
                            np.ravel(sim.points[-(num_points - num_fixed_points):, :])))

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_all_marginal1(self):
        num_points = 2000
        num_fixed_points = num_points
        num_free_points = 0

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=251)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.ravel(sim.landmarks[sim.observed_landmarks, :])

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())

    def test_all_marginal2(self):
        num_points = 2000
        num_fixed_points = num_points
        num_free_points = 0

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=num_points, seed=252)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, b = fg.observation_system
        x = np.ravel(sim.landmarks[sim.observed_landmarks, :])

        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), b).all())


class TestOdometrySystem(unittest.TestCase):

    def test_no_marginal1(self):
        sim = new_simulation(point_dim=1, landmark_dim=1, seed=311)
        fg = sim_to_factorgraph(sim)

        A, t = fg.odometry_system
        x = np.ravel(sim.points)

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_no_marginal2(self):
        sim = new_simulation(point_dim=3, landmark_dim=1, seed=312)
        fg = sim_to_factorgraph(sim)

        A, t = fg.odometry_system
        x = np.ravel(sim.points)

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_no_marginal3(self):
        sim = new_simulation(point_dim=1, landmark_dim=3, seed=313)
        fg = sim_to_factorgraph(sim)

        A, t = fg.odometry_system
        x = np.ravel(sim.points)

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_no_marginal4(self):
        sim = new_simulation(point_dim=3, landmark_dim=3, seed=314)
        fg = sim_to_factorgraph(sim)

        A, t = fg.odometry_system
        x = np.ravel(sim.points)

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_marginal1(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=1, landmark_dim=1, num_points=num_points, seed=321)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system
        x = np.ravel(sim.points[-num_free_points:, :])

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_marginal2(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=322)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system
        x = np.ravel(sim.points[-num_free_points:, :])

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_marginal3(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=323)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system
        x = np.ravel(sim.points[-num_free_points:, :])

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_marginal4(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=num_points, seed=324)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system
        x = np.ravel(sim.points[-num_free_points:, :])

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_over_marginal1(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=1, num_points=num_points, seed=331)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system
        x = np.ravel(sim.points[-(num_points - num_fixed_points):, :])

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_over_marginal2(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=332)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system
        x = np.ravel(sim.points[-(num_points - num_fixed_points):, :])

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_over_marginal3(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=333)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system
        x = np.ravel(sim.points[-(num_points - num_fixed_points):, :])

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_over_marginal4(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=334)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system
        x = np.ravel(sim.points[-(num_points - num_fixed_points):, :])

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_under_marginal1(self):
        num_points = 2000
        num_fixed_points = 100
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=1, num_points=num_points, seed=341)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system
        x = np.ravel(sim.points[-(num_points - num_fixed_points):, :])

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_under_marginal2(self):
        num_points = 2000
        num_fixed_points = 242
        num_free_points = 869

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=342)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system
        x = np.ravel(sim.points[-(num_points - num_fixed_points):, :])

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_under_marginal3(self):
        num_points = 2000
        num_fixed_points = 87
        num_free_points = 419

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=343)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system
        x = np.ravel(sim.points[-(num_points - num_fixed_points):, :])

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_under_marginal4(self):
        num_points = 2000
        num_fixed_points = 111
        num_free_points = 642

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=num_points, seed=344)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system
        x = np.ravel(sim.points[-(num_points - num_fixed_points):, :])

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], x.size)
        self.assertTrue(np.isclose(A.dot(x), t).all())

    def test_all_marginal1(self):
        num_points = 2000
        num_fixed_points = num_points
        num_free_points = 0

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=351)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], 0)

    def test_all_marginal2(self):
        num_points = 2000
        num_fixed_points = num_points
        num_free_points = 0

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=num_points, seed=352)
        fg = sim_to_factorgraph(sim)
        fg.free_point_window = num_free_points

        points = [x for x in fg.variables if isinstance(x, variable.PointVariable)]
        for i in range(num_fixed_points):
            points[i].position = sim.points[i, :]

        A, t = fg.odometry_system

        self.assertEqual(A.shape[0], t.size)
        self.assertEqual(A.shape[1], 0)


class TestAccessSpeed(unittest.TestCase):

    def test_observation_speed(self):

        sim = new_simulation(seed=2)
        fg = factorgraph.GaussianFactorGraph()

        point_variables = [variable.PointVariable(sim.point_dim) for _ in range(sim.num_points)]
        landmark_variables = [variable.LandmarkVariable(sim.landmark_dim, sim.landmark_labels[i])
                              for i in range(sim.num_landmarks)]

        odometry_factors = [factor.OdometryFactor(point_variables[u], point_variables[v], R, t)
                            for (u, v), R, t in zip(*sim.odometry_factors())]
        observation_factors = [factor.ObservationFactor(point_variables[u], landmark_variables[v], H, d)
                               for (u, v), H, d in zip(*sim.observation_factors())]

        i = 0
        j = 0
        for pv in point_variables:

            if pv == odometry_factors[i].head:
                fg.add_factor(odometry_factors[i])
                i += 1

            while j < len(observation_factors) and pv == observation_factors[j].tail:
                fg.add_factor(observation_factors[j])
                j += 1

            fg.observation_system

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
