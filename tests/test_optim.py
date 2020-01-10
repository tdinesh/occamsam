import unittest

import numpy as np
import scipy as sp
import scipy.sparse

import optim
import factorgraph
from simulator import new_simulation

import matplotlib.pyplot as plt


class TestLeastSquares(unittest.TestCase):

    def test_no_noise1(self):

        sim = new_simulation(point_dim=3, landmark_dim=1, seed=9, observation_noise=0.0, odometry_noise=0.0)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.LeastSquares(fg, verbosity=True)
        optimizer.optimize()
        optimizer.update()

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        m, p = np.ravel(sim.landmark_positions), np.ravel(sim.point_positions)

        self.assertTrue(np.allclose(m, m_hat, atol=1e-4))
        self.assertTrue(np.allclose(p, p_hat, atol=1e-3))

    def test_no_noise2(self):

        sim = new_simulation(point_dim=3, landmark_dim=3, seed=6, observation_noise=0.0, odometry_noise=0.0)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.LeastSquares(fg, verbosity=True)
        optimizer.optimize()
        optimizer.update()

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        m, p = np.ravel(sim.landmark_positions), np.ravel(sim.point_positions)

        self.assertTrue(np.allclose(m, m_hat, atol=1e-4))
        self.assertTrue(np.allclose(p, p_hat, atol=1e-3))

    def test_noise1(self):

        observation_noise = 0.05
        odometry_noise = 0.06

        sim = new_simulation(point_dim=3, landmark_dim=1, seed=11, observation_noise=observation_noise, odometry_noise=odometry_noise)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.LeastSquares(fg, verbosity=True)
        optimizer.optimize()
        optimizer.update()

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        m, p = np.ravel(sim.landmark_positions), np.ravel(sim.point_positions)

        mu_m_hat = np.mean(m - m_hat)
        sigma_m_hat = np.sqrt(np.linalg.norm(m - m_hat)**2 / len(m))
        mu_p_hat = np.mean(p - p_hat)
        sigma_p_hat = np.sqrt(np.linalg.norm(p - p_hat)**2 / len(p))

        self.assertTrue(np.linalg.norm(optimizer.res_d)**2 < len(optimizer.res_d) * 9 * observation_noise**2)
        self.assertTrue(np.linalg.norm(optimizer.res_t)**2 < len(optimizer.res_t) * 9 * odometry_noise**2)

    def test_noise2(self):

        observation_noise = 0.05
        odometry_noise = 0.06

        sim = new_simulation(point_dim=3, landmark_dim=3, seed=5, observation_noise=observation_noise, odometry_noise=odometry_noise)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.LeastSquares(fg, verbosity=True)
        optimizer.optimize()
        optimizer.update()

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        m, p = np.ravel(sim.landmark_positions), np.ravel(sim.point_positions)

        mu_m_hat = np.mean(m - m_hat)
        sigma_m_hat = np.sqrt(np.linalg.norm(m - m_hat)**2 / len(m))
        mu_p_hat = np.mean(p - p_hat)
        sigma_p_hat = np.sqrt(np.linalg.norm(p - p_hat)**2 / len(p))

        self.assertTrue(np.linalg.norm(optimizer.res_d)**2 < len(optimizer.res_d) * 9 * observation_noise**2)
        self.assertTrue(np.linalg.norm(optimizer.res_t)**2 < len(optimizer.res_t) * 9 * odometry_noise**2)


class TestWeightedLeastSquares(unittest.TestCase):

    def test_no_noise1(self):

        sim = new_simulation(point_dim=3, landmark_dim=1, seed=21)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.WeightedLeastSquares(fg)
        optimizer.optimize()
        optimizer.update()

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        m, p = np.ravel(sim.landmark_positions), np.ravel(sim.point_positions)

        self.assertTrue(np.allclose(m, m_hat, atol=1e-4))
        self.assertTrue(np.allclose(p, p_hat, atol=1e-3))

    def test_no_noise2(self):

        sim = new_simulation(point_dim=3, landmark_dim=3, seed=23)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.WeightedLeastSquares(fg, verbosity=True)
        optimizer.optimize()
        optimizer.update()

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        m, p = np.ravel(sim.landmark_positions), np.ravel(sim.point_positions)

        self.assertTrue(np.allclose(m, m_hat, atol=1e-4))
        self.assertTrue(np.allclose(p, p_hat, atol=1e-3))

    def test_noise1(self):
        observation_noise = 0.01
        odometry_noise = 0.02

        sim = new_simulation(point_dim=3, landmark_dim=1, seed=11, observation_noise=observation_noise,
                             odometry_noise=odometry_noise, noise_matrix='diag')
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.WeightedLeastSquares(fg)
        optimizer.optimize()
        optimizer.update()

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        m, p = np.ravel(sim.landmark_positions), np.ravel(sim.point_positions)

        mu_m_hat = np.mean(m - m_hat)
        sigma_m_hat = np.sqrt(np.linalg.norm(m - m_hat) ** 2 / len(m))
        mu_p_hat = np.mean(p - p_hat)
        sigma_p_hat = np.sqrt(np.linalg.norm(p - p_hat) ** 2 / len(p))

        self.assertTrue(np.linalg.norm(optimizer.res_d) ** 2 < len(optimizer.res_d) * 4 * observation_noise ** 2)
        self.assertTrue(np.linalg.norm(optimizer.res_t) ** 2 < len(optimizer.res_t) * 4 * odometry_noise ** 2)

    def test_noise2(self):
        observation_noise = 0.05
        odometry_noise = 0.06

        sim = new_simulation(point_dim=3, landmark_dim=3, seed=11, observation_noise=observation_noise,
                             odometry_noise=odometry_noise, noise_matrix='diag')
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.WeightedLeastSquares(fg)
        optimizer.optimize()
        optimizer.update()

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        m, p = np.ravel(sim.landmark_positions), np.ravel(sim.point_positions)

        mu_m_hat = np.mean(m - m_hat)
        sigma_m_hat = np.sqrt(np.linalg.norm(m - m_hat) ** 2 / len(m))
        mu_p_hat = np.mean(p - p_hat)
        sigma_p_hat = np.sqrt(np.linalg.norm(p - p_hat) ** 2 / len(p))

        self.assertTrue(np.linalg.norm(optimizer.res_d) ** 2 < len(optimizer.res_d) * 4 * observation_noise ** 2)
        self.assertTrue(np.linalg.norm(optimizer.res_t) ** 2 < len(optimizer.res_t) * 4 * odometry_noise ** 2)

    def test_noise3(self):

        observation_noise = 0.01
        odometry_noise = 0.02

        sim = new_simulation(point_dim=3, landmark_dim=1, seed=17, observation_noise=observation_noise,
                             odometry_noise=odometry_noise, noise_matrix='diag')
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.WeightedLeastSquares(fg)
        optimizer.optimize()
        optimizer.update()

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        m, p = np.ravel(sim.landmark_positions), np.ravel(sim.point_positions)

        mu_m_hat = np.mean(m - m_hat)
        sigma_m_hat = np.sqrt(np.linalg.norm(m - m_hat) ** 2 / len(m))
        mu_p_hat = np.mean(p - p_hat)
        sigma_p_hat = np.sqrt(np.linalg.norm(p - p_hat) ** 2 / len(p))

        self.assertTrue(np.linalg.norm(optimizer.res_d) ** 2 < len(optimizer.res_d) * 4 * observation_noise ** 2)
        self.assertTrue(np.linalg.norm(optimizer.res_t) ** 2 < len(optimizer.res_t) * 4 * odometry_noise ** 2)

    def test_sequential1(self):

        observation_noise = 0.01
        odometry_noise = 0.02

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=100, seed=266, observation_noise=observation_noise,
                             odometry_noise=odometry_noise, noise_matrix='diag')

        num_partitions = 10
        partition_indptr = np.sort(
            np.concatenate([[0], np.random.choice(sim.num_points, num_partitions - 1), [sim.num_points]]))

        fpw = int(np.max(partition_indptr[1:] - partition_indptr[:-1]) + 5)
        fg = factorgraph.GaussianFactorGraph(free_point_window=fpw)

        optimizer = optim.WeightedLeastSquares(fg)
        for i in range(num_partitions):

            for f in sim.factors((partition_indptr[i], partition_indptr[i+1])):
                fg.add_factor(f)

            optimizer.optimize()
            optimizer.update()

            p_hat = np.concatenate([p.position for p in fg.points])
            p = np.ravel(sim.point_positions[:partition_indptr[i + 1], :])

            self.assertTrue(np.linalg.norm(optimizer.res_d) ** 2 < len(optimizer.res_d) * 4 * observation_noise ** 2)
            self.assertTrue(np.linalg.norm(optimizer.res_t) ** 2 < len(optimizer.res_t) * 4 * odometry_noise ** 2)
            self.assertTrue(np.allclose(p, p_hat, atol=0.1))

    def test_sequential2(self):

        observation_noise = 0.01
        odometry_noise = 0.02

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=100, seed=268, observation_noise=observation_noise,
                             odometry_noise=odometry_noise, noise_matrix='diag')

        num_partitions = 10
        partition_indptr = np.sort(
            np.concatenate([[0], np.random.choice(sim.num_points, num_partitions - 1), [sim.num_points]]))

        fpw = int(np.max(partition_indptr[1:] - partition_indptr[:-1]) + 5)
        fg = factorgraph.GaussianFactorGraph(free_point_window=fpw)

        optimizer = optim.WeightedLeastSquares(fg)
        for i in range(num_partitions):

            for f in sim.factors((partition_indptr[i], partition_indptr[i+1])):
                fg.add_factor(f)

            optimizer.optimize()
            optimizer.update()

            p_hat = np.concatenate([p.position for p in fg.points])
            p = np.ravel(sim.point_positions[:partition_indptr[i + 1], :])

            self.assertTrue(np.linalg.norm(optimizer.res_d) ** 2 < len(optimizer.res_d) * 4 * observation_noise ** 2)
            self.assertTrue(np.linalg.norm(optimizer.res_t) ** 2 < len(optimizer.res_t) * 4 * odometry_noise ** 2)
            self.assertTrue(np.allclose(p, p_hat, atol=0.1))


class TestOccam(unittest.TestCase):

    def test_no_noise1(self):

        sim = new_simulation(point_dim=3, landmark_dim=1, seed=16)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        landmarks = fg.landmarks
        self.assertTrue(sim.num_unique_landmarks < len(landmarks))

        optimizer = optim.Occam(fg)
        optimizer.optimize()
        optimizer.update()

        unique_landmarks = fg.landmarks
        self.assertEqual(sim.num_unique_landmarks, len(unique_landmarks))

        sim_groups, _ = sim.equivalences()
        fg_groups = set(frozenset(g) for g in fg.correspondence_map.set_map().values())
        diff = sim_groups.symmetric_difference(fg_groups)
        self.assertTrue(len(diff) == 0)

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        unique_order = [sim.unique_index_map[landmarks.index(k)] for k in unique_landmarks]
        m, p = np.ravel(sim.unique_landmark_positions[unique_order, :]), np.ravel(sim.point_positions)

        self.assertTrue(np.allclose(p, p_hat))
        self.assertTrue(np.allclose(m, m_hat))

    def test_no_noise2(self):

        sim = new_simulation(point_dim=3, landmark_dim=3, seed=42)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        landmarks = fg.landmarks
        self.assertTrue(sim.num_unique_landmarks < len(landmarks))

        optimizer = optim.Occam(fg)
        optimizer.optimize()
        optimizer.update()

        unique_landmarks = fg.landmarks
        self.assertEqual(sim.num_unique_landmarks, len(unique_landmarks))

        sim_groups, _ = sim.equivalences()
        fg_groups = set(frozenset(g) for g in fg.correspondence_map.set_map().values())
        diff = sim_groups.symmetric_difference(fg_groups)
        self.assertTrue(len(diff) == 0)

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        unique_order = [sim.unique_index_map[landmarks.index(k)] for k in unique_landmarks]
        m, p = np.ravel(sim.unique_landmark_positions[unique_order, :]), np.ravel(sim.point_positions)

        self.assertTrue(np.allclose(p, p_hat))
        self.assertTrue(np.allclose(m, m_hat))

    def test_noise1(self):
        observation_noise = 0.01
        odometry_noise = 0.02

        # sim = new_simulation(point_dim=3, landmark_dim=1, seed=11, observation_noise=observation_noise,
        #                      odometry_noise=odometry_noise, noise_matrix='diag')
        sim = new_simulation(point_dim=3, landmark_dim=1, seed=17, observation_noise=observation_noise,
                             odometry_noise=odometry_noise, noise_matrix='diag')
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.Occam(fg)
        optimizer.optimize()
        optimizer.update()

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        m, p = np.ravel(sim.landmark_positions), np.ravel(sim.point_positions)

        self.assertTrue(np.linalg.norm(optimizer.res_d) ** 2 < len(optimizer.res_d) * 4 * observation_noise ** 2)
        self.assertTrue(np.linalg.norm(optimizer.res_t) ** 2 < len(optimizer.res_t) * 4 * odometry_noise ** 2)
        self.assertTrue(np.allclose(p, p_hat, atol=0.1))

    def test_noise2(self):
        observation_noise = 0.01
        odometry_noise = 0.02

        sim = new_simulation(point_dim=3, landmark_dim=3, seed=20, observation_noise=observation_noise,
                             odometry_noise=odometry_noise, noise_matrix='diag')
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.Occam(fg)
        optimizer.optimize()
        optimizer.update()

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        m, p = np.ravel(sim.landmark_positions), np.ravel(sim.point_positions)

        self.assertTrue(np.linalg.norm(optimizer.res_d) ** 2 < len(optimizer.res_d) * 4 * observation_noise ** 2)
        self.assertTrue(np.linalg.norm(optimizer.res_t) ** 2 < len(optimizer.res_t) * 4 * odometry_noise ** 2)
        self.assertTrue(np.allclose(p, p_hat, atol=0.1))

    def test_sequential1(self):

        observation_noise = 0.01
        odometry_noise = 0.01

        sim = new_simulation(point_dim=3, landmark_dim=3, num_points=1000, seed=66, observation_noise=observation_noise,
                             odometry_noise=odometry_noise)

        num_partitions = 90
        partition_indptr = np.sort(
            np.concatenate([[0], np.random.choice(sim.num_points, num_partitions - 1), [sim.num_points]]))

        fpw = np.max(partition_indptr[1:] - partition_indptr[:-1])
        fg = factorgraph.GaussianFactorGraph(free_point_window=fpw)

        for i in range(num_partitions):

            for f in sim.factors((partition_indptr[i], partition_indptr[i+1])):
                fg.add_factor(f)

            landmarks = fg.landmarks
            self.assertTrue(sim.num_unique_landmarks < len(landmarks))

            optimizer = optim.Occam(fg)
            optimizer.optimize()
            optimizer.update()

            unique_landmarks = fg.landmarks
            self.assertEqual(sim.num_unique_landmarks, len(unique_landmarks))

            sim_groups, _ = sim.equivalences()
            fg_groups = set(frozenset(g) for g in fg.correspondence_map.set_map().values())
            diff = sim_groups.symmetric_difference(fg_groups)
            self.assertTrue(len(diff) == 0)

            m_hat = np.concatenate([m.position for m in fg.landmarks])
            p_hat = np.concatenate([p.position for p in fg.points])

            unique_order = [sim.unique_index_map[landmarks.index(k)] for k in unique_landmarks]
            m, p = np.ravel(sim.unique_landmark_positions[unique_order, :]), np.ravel(sim.point_positions)

            self.assertTrue(np.allclose(p, p_hat))
            self.assertTrue(np.allclose(m, m_hat))

    def test_sequential_no_noise(self):

        num_points = 1000

        sim = new_simulation(point_dim=3, landmark_dim=1, num_points=num_points, seed=66)

        num_partitions = 10
        # partition_indptr = np.sort(
        #     np.concatenate([[0], np.random.choice(sim.num_points, num_partitions - 1), [sim.num_points]]))
        partition_indptr = np.linspace(0, num_points, num_partitions + 1, dtype=np.int)

        fpw = int(np.max(partition_indptr[1:] - partition_indptr[:-1]) + 5)
        fg = factorgraph.GaussianFactorGraph(free_point_window=fpw)

        landmarks = sim.landmark_variables

        optimizer = optim.Occam(fg)
        for i in range(num_partitions):

            for f in sim.factors((partition_indptr[i], partition_indptr[i+1])):
                fg.add_factor(f)

            optimizer.optimize()
            optimizer.update()

            unique_landmarks = fg.landmarks
            unique_order = [sim.unique_index_map[landmarks.index(k)] for k in unique_landmarks]

            sim_groups, _ = sim.equivalences((0, partition_indptr[i+1]))
            fg_groups = set(frozenset(g) for g in fg.correspondence_map.set_map().values())
            diff = sim_groups.symmetric_difference(fg_groups)
            self.assertTrue(len(diff) == 0)

            p_hat = np.concatenate([p.position for p in fg.points])
            m_hat = np.concatenate([m.position for m in fg.landmarks])

            m = np.ravel(sim.unique_landmark_positions[unique_order, :])
            p = np.ravel(sim.point_positions[:partition_indptr[i + 1], :])

            self.assertTrue(np.allclose(p, p_hat, atol=0.1))
            self.assertTrue(np.allclose(m, m_hat))


if __name__ == '__main__':
    unittest.main()
