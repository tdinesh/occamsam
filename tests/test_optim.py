import unittest

import numpy as np

import optim
import factorgraph
from simulator import new_simulation


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

        self.assertTrue(np.linalg.norm(optimizer.res_d) ** 2 < len(optimizer.res_d) * 9 * observation_noise ** 2)
        self.assertTrue(np.linalg.norm(optimizer.res_t) ** 2 < len(optimizer.res_t) * 9 * odometry_noise ** 2)

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

        self.assertTrue(np.linalg.norm(optimizer.res_d) ** 2 < len(optimizer.res_d) * 9 * observation_noise ** 2)
        self.assertTrue(np.linalg.norm(optimizer.res_t) ** 2 < len(optimizer.res_t) * 9 * odometry_noise ** 2)


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

    def test_noise(self):
        observation_noise = 0.01
        odometry_noise = 0.02

        # 11
        sim = new_simulation(point_dim=3, landmark_dim=1, seed=16, observation_noise=observation_noise,
                             odometry_noise=odometry_noise, noise_matrix='diag')
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


if __name__ == '__main__':
    unittest.main()
