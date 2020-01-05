import unittest

import numpy as np

import optim
import factorgraph
from simulator import new_simulation


class TestLeastSquares(unittest.TestCase):

    def test_no_noise(self):

        sim = new_simulation(point_dim=3, landmark_dim=1, seed=9, observation_noise=0.0, odometry_noise=0.0)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.LeastSquares(fg)
        optimizer.optimize()
        optimizer.update()

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        m, p = np.ravel(sim.landmark_positions), np.ravel(sim.point_positions)

        self.assertTrue(np.allclose(m, m_hat, atol=1e-4))
        self.assertTrue(np.allclose(p, p_hat, atol=1e-3))

    def test_noise(self):

        observation_noise = 0.05
        odometry_noise = 0.06

        sim = new_simulation(point_dim=3, landmark_dim=1, seed=11, observation_noise=observation_noise, odometry_noise=odometry_noise)
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.LeastSquares(fg)
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

    def test_noise(self):
        observation_noise = 0.05
        odometry_noise = 0.06

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


class TestOccam(unittest.TestCase):

    def test_noise(self):
        observation_noise = 0.05
        odometry_noise = 0.06

        sim = new_simulation(point_dim=3, landmark_dim=1, seed=11, observation_noise=observation_noise,
                             odometry_noise=odometry_noise, noise_matrix='diag')
        fg = factorgraph.GaussianFactorGraph()
        for f in sim.factors():
            fg.add_factor(f)

        optimizer = optim.WeightedLeastSquares(fg)
        optimizer.optimize()
        optimizer.update()

        optimizer = optim.Occam(fg)
        optimizer.optimize()
        optimizer.update()

        m_hat = np.concatenate([m.position for m in fg.landmarks])
        p_hat = np.concatenate([p.position for p in fg.points])

        m, p = np.ravel(sim.landmark_positions), np.ravel(sim.point_positions)

        print('yes')


if __name__ == '__main__':
    unittest.main()
