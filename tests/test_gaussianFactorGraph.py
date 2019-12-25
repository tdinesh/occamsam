import unittest
import numpy as np

import factorgraph
from simulator import new_simulation


class TestObservationArray(unittest.TestCase):

    def test_observation_array1(self):
        sim = new_simulation(point_dim=1, landmark_dim=1, seed=111)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        _, d_sys = fg.observation_system()
        d_sim = np.concatenate([d for _, _, d in zip(*sim.observation_measurements())])

        self.assertEqual(len(d_sys), len(d_sim))
        self.assertTrue(np.allclose(d_sys, d_sim))

    def test_observation_array2(self):
        sim = new_simulation(point_dim=3, landmark_dim=3, seed=112)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        _, d_sys = fg.observation_system()
        d_sim = np.concatenate([d for _, _, d in zip(*sim.observation_measurements())])

        self.assertEqual(len(d_sys), len(d_sim))
        self.assertTrue(np.allclose(d_sys, d_sim))

    def test_observation_array3(self):
        sim = new_simulation(point_dim=3, landmark_dim=1, seed=113)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        _, d_sys = fg.observation_system()
        d_sim = np.concatenate([d for _, _, d in zip(*sim.observation_measurements())])

        self.assertEqual(len(d_sys), len(d_sim))
        self.assertTrue(np.allclose(d_sys, d_sim))

    def test_observation_array4(self):
        sim = new_simulation(point_dim=1, landmark_dim=3, seed=114)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        _, d_sys = fg.observation_system()
        d_sim = np.concatenate([d for _, _, d in zip(*sim.observation_measurements())])

        self.assertEqual(len(d_sys), len(d_sim))
        self.assertTrue(np.allclose(d_sys, d_sim))


class TestOdometryArray(unittest.TestCase):

    def test_odometry_array1(self):
        sim = new_simulation(point_dim=1, landmark_dim=1, seed=121)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        _, t_sys = fg.odometry_system()
        t_sim = np.concatenate([sim.point_positions[0, :]] + [np.dot(R, t) for _, R, t in zip(*sim.odometry_measurements())])

        self.assertEqual(len(t_sys), len(t_sim))
        self.assertTrue(np.allclose(t_sys, t_sim))

    def test_odometry_array2(self):
        sim = new_simulation(point_dim=3, landmark_dim=3, seed=122)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        _, t_sys = fg.odometry_system()
        t_sim = np.concatenate([sim.point_positions[0, :]] + [np.dot(R, t) for _, R, t in zip(*sim.odometry_measurements())])

        self.assertEqual(len(t_sys), len(t_sim))
        self.assertTrue(np.allclose(t_sys, t_sim))

    def test_odometry_array3(self):
        sim = new_simulation(point_dim=3, landmark_dim=1, seed=123)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        _, t_sys = fg.odometry_system()
        t_sim = np.concatenate([sim.point_positions[0, :]] + [np.dot(R, t) for _, R, t in zip(*sim.odometry_measurements())])

        self.assertEqual(len(t_sys), len(t_sim))
        self.assertTrue(np.allclose(t_sys, t_sim))

    def test_odometry_array4(self):
        sim = new_simulation(point_dim=1, landmark_dim=3, seed=124)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        _, t_sys = fg.odometry_system()
        t_sim = np.concatenate([sim.point_positions[0, :]] + [np.dot(R, t) for _, R, t in zip(*sim.odometry_measurements())])

        self.assertEqual(len(t_sys), len(t_sim))
        self.assertTrue(np.allclose(t_sys, t_sim))


class TestObservationSystem(unittest.TestCase):

    def test_no_marginal1(self):
        sim = new_simulation(point_dim=1, landmark_dim=1, seed=211)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions)

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_no_marginal2(self):
        sim = new_simulation(point_dim=3, landmark_dim=1, seed=212)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions)

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_no_marginal3(self):
        sim = new_simulation(point_dim=1, landmark_dim=3, seed=213)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions)

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_no_marginal4(self):
        sim = new_simulation(point_dim=3, landmark_dim=3, seed=214)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions)

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_marginal_small(self):
        num_points = 100
        num_fixed_points = 15
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=1, landmark_dim=1, num_points=num_points, num_landmarks=20, seed=221)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions[-num_free_points:, :])

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_marginal1(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=1, landmark_dim=1, num_points=num_points, seed=221)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions[-num_free_points:, :])

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_marginal2(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=222)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions[-num_free_points:, :])

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_marginal3(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=223)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions[-num_free_points:, :])

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_marginal4(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=num_points, seed=224)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions[-num_free_points:, :])

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_over_marginal1(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=1, num_points=num_points, seed=231)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_over_marginal2(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=232)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_over_marginal3(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=233)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_over_marginal4(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=num_points, seed=234)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_under_marginal1(self):
        num_points = 2000
        num_fixed_points = 100
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=1, num_points=num_points, seed=241)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_under_marginal2(self):
        num_points = 2000
        num_fixed_points = 242
        num_free_points = 869

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=242)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_under_marginal3(self):
        num_points = 2000
        num_fixed_points = 87
        num_free_points = 419

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=243)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_under_marginal4(self):
        num_points = 2000
        num_fixed_points = 111
        num_free_points = 642

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=num_points, seed=244)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)
        p = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size + p.size, Am.shape[1] + Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m) + Ap.dot(p), d))

    def test_all_marginal1(self):
        num_points = 2000
        num_fixed_points = num_points
        num_free_points = 0

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=251)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size, Am.shape[1])
        self.assertEqual(0, Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m), d))

    def test_all_marginal2(self):
        num_points = 2000
        num_fixed_points = num_points
        num_free_points = 0

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=num_points, seed=252)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        Am, Ap, d = fg.observation_system()
        m = np.ravel(sim.landmark_positions)

        self.assertEqual(d.size, Am.shape[0])
        self.assertEqual(d.size, Ap.shape[0])
        self.assertEqual(m.size, Am.shape[1])
        self.assertEqual(0, Ap.shape[1])
        self.assertTrue(np.allclose(Am.dot(m), d))


class TestOdometrySystem(unittest.TestCase):

    def test_no_marginal1(self):
        sim = new_simulation(point_dim=1, landmark_dim=1, seed=311)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions)

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_no_marginal2(self):
        sim = new_simulation(point_dim=3, landmark_dim=1, seed=312)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions)

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_no_marginal3(self):
        sim = new_simulation(point_dim=1, landmark_dim=3, seed=313)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions)

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_no_marginal4(self):
        sim = new_simulation(point_dim=3, landmark_dim=3, seed=314)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions)

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_marginal1(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=1, landmark_dim=1, num_points=num_points, seed=321)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions[-num_free_points:, :])

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_marginal2(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=322)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions[-num_free_points:, :])

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_marginal3(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=323)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions[-num_free_points:, :])

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_marginal4(self):
        num_points = 2000
        num_fixed_points = 1500
        num_free_points = num_points - num_fixed_points

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=num_points, seed=324)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions[-num_free_points:, :])

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_over_marginal1(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=1, num_points=num_points, seed=331)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_over_marginal2(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=332)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_over_marginal3(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=333)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_over_marginal4(self):
        num_points = 2000
        num_fixed_points = 1200
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=334)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_under_marginal1(self):
        num_points = 2000
        num_fixed_points = 100
        num_free_points = 500

        sim = new_simulation(point_dim=1, landmark_dim=1, num_points=num_points, seed=341)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_under_marginal2(self):
        num_points = 2000
        num_fixed_points = 242
        num_free_points = 869

        sim = new_simulation(point_dim=1, landmark_dim=3, num_points=num_points, seed=342)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_under_marginal3(self):
        num_points = 2000
        num_fixed_points = 87
        num_free_points = 419

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=343)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_under_marginal4(self):
        num_points = 2000
        num_fixed_points = 111
        num_free_points = 642

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=num_points, seed=344)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()
        x = np.ravel(sim.point_positions[-(num_points - num_fixed_points):, :])

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_all_marginal1(self):
        num_points = 2000
        num_fixed_points = num_points
        num_free_points = 0

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=351)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(0, A.shape[1])

    def test_all_marginal2(self):
        num_points = 2000
        num_fixed_points = num_points
        num_free_points = 0

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=num_points, seed=352)
        fg = factorgraph.GaussianFactorGraph(num_free_points)
        for f in sim.fix_points(list(range(num_fixed_points))).factors():
            fg.add_factor(f)

        A, b = fg.odometry_system()

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(0, A.shape[1])


class TestMerge(unittest.TestCase):

    def test_merge_once(self):
        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=30, num_landmarks=30, seed=410)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        landmarks = sim.landmark_variables
        equiv_groups, equiv_pairs = sim.equivalences()

        fg._merge_landmarks(equiv_pairs)
        unique_landmarks = fg.landmarks
        unique_order = [sim.unique_index_map[landmarks.index(k)] for k in unique_landmarks]

        self.assertEqual(len(fg.correspondence_map.set_map().keys()), sim.num_unique_landmarks)
        fg_groups = set(frozenset(g) for g in fg.correspondence_map.set_map().values())
        diff = equiv_groups.symmetric_difference(fg_groups)
        self.assertTrue(len(diff) == 0)

        A, b = fg.observation_system()
        x = np.concatenate((np.ravel(sim.unique_landmark_positions[unique_order, :]), np.ravel(sim.point_positions)))

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_merge_once_big(self):
        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=2000, num_landmarks=120, seed=411)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        landmarks = sim.landmark_variables

        equiv_groups, equiv_pairs = sim.equivalences()

        fg._merge_landmarks(equiv_pairs)
        unique_landmarks = fg.landmarks
        unique_order = [sim.unique_index_map[landmarks.index(k)] for k in unique_landmarks]

        self.assertEqual(len(fg.correspondence_map.set_map().keys()), sim.num_unique_landmarks)
        fg_groups = set(frozenset(g) for g in fg.correspondence_map.set_map().values())
        diff = equiv_groups.symmetric_difference(fg_groups)
        self.assertTrue(len(diff) == 0)

        A, b = fg.observation_system()
        x = np.concatenate((np.ravel(sim.unique_landmark_positions[unique_order, :]), np.ravel(sim.point_positions)))

        self.assertEqual(b.size, A.shape[0])
        self.assertEqual(x.size, A.shape[1])
        self.assertTrue(np.allclose(A.dot(x), b))

    def test_merge_sequence(self):
        sim = new_simulation(point_dim=1, landmark_dim=3, seed=412)
        fg = factorgraph.GaussianFactorGraph()

        landmarks = sim.landmark_variables

        num_partitions = 5
        partition_indptr = np.sort(np.concatenate([[0], np.random.choice(sim.num_points, num_partitions-1), [sim.num_points]]))

        for i in range(num_partitions):

            for f in sim.factors((partition_indptr[i], partition_indptr[i+1])):
                fg.add_factor(f)

            equiv_groups, equiv_pairs = sim.equivalences((0, partition_indptr[i+1]))

            fg._merge_landmarks(equiv_pairs)

            unique_landmarks = fg.landmarks
            unique_order = [sim.unique_index_map[landmarks.index(k)] for k in unique_landmarks]

            self.assertEqual(len(fg.correspondence_map.set_map().keys()), len(unique_landmarks))
            fg_groups = set(frozenset(g) for g in fg.correspondence_map.set_map().values())
            diff = equiv_groups.symmetric_difference(fg_groups)
            self.assertTrue(len(diff) == 0)

            A, b = fg.observation_system()
            x = np.concatenate((np.ravel(sim.unique_landmark_positions[unique_order, :]),
                                np.ravel(sim.point_positions[:partition_indptr[i+1], :])))

            self.assertEqual(b.size, A.shape[0])
            self.assertEqual(x.size, A.shape[1])
            self.assertTrue(np.allclose(A.dot(x), b))


class TestAccessSpeed(unittest.TestCase):

    def test_observation_speed(self):

        sim = new_simulation(seed=2, point_dim=3, landmark_dim=1)
        fg = factorgraph.GaussianFactorGraph(free_point_window=10)
        for f in sim.factors():
            f.add_factor(f)
            A, b = fg.observation_system()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
