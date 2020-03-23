import numpy as np

import time

import optim
import factorgraph
from simulator import new_simulation

NUM_OBSERVATIONS = 4
NUM_TRIALS = 40

NUM_POINTS = 1000
NUM_LANDMARKS = 80

def observed_landmark_index(fg, sim):
    return np.array([sim.landmark_variables.index(m) for m in fg.landmarks])


def em_trial(seed=12, max_observations=4):

    observation_noise = 0.01
    odometry_noise = 0.02

    sim = new_simulation(point_dim=3, landmark_dim=1,
                         num_points=NUM_POINTS,
                         num_landmarks=NUM_LANDMARKS,
                         seed=seed,
                         observation_noise=observation_noise,
                         odometry_noise=odometry_noise,
                         noise_matrix='identity')

    fg = factorgraph.GaussianFactorGraph()
    for f in sim.factors(max_observations=max_observations):
        fg.add_factor(f)

    start = time.time()
    optimizer = optim.EM(fg)
    optimizer.optimize()
    optimizer.update()
    end = time.time()
    time_elapsed = end - start

    m_hat = np.concatenate([m.position for m in fg.landmarks])
    p_hat = np.concatenate([p.position for p in fg.points])

    m, p = np.ravel(sim.landmark_positions[observed_landmark_index(fg, sim), :]), np.ravel(sim.point_positions)

    me = np.mean(np.concatenate([(m - m_hat), (p - p_hat)]))

    return me, time_elapsed, optimizer.iter_counter


def occam_trial(seed=12, max_observations=4):

    observation_noise = 0.01
    odometry_noise = 0.02

    sim = new_simulation(point_dim=3, landmark_dim=1,
                         num_points=NUM_POINTS,
                         num_landmarks=NUM_LANDMARKS,
                         seed=seed,
                         observation_noise=observation_noise,
                         odometry_noise=odometry_noise,
                         noise_matrix='identity')

    fg = factorgraph.GaussianFactorGraph()
    for f in sim.factors(max_observations=max_observations):
        fg.add_factor(f)

    start = time.time()
    optimizer = optim.Occam(fg)
    optimizer.optimize()
    optimizer.update(merge=False)
    end = time.time()
    time_elapsed = end - start

    m_hat = np.concatenate([m.position for m in fg.landmarks])
    p_hat = np.concatenate([p.position for p in fg.points])

    m, p = np.ravel(sim.landmark_positions[observed_landmark_index(fg, sim), :]), np.ravel(sim.point_positions)

    me = np.mean(np.concatenate([(m - m_hat), (p - p_hat)]))

    return me, time_elapsed


if __name__ == '__main__':

    np.random.seed(2021)
    seeds = np.random.permutation(np.arange(NUM_TRIALS))

    occam_data = np.zeros((NUM_OBSERVATIONS, NUM_TRIALS, 2))
    em_data = np.zeros((NUM_OBSERVATIONS, NUM_TRIALS, 3))

    for i, r in enumerate(range(1, NUM_OBSERVATIONS+1)):

        for j, s in enumerate(seeds):

            print('Max Obsv. = {:}, Trial = {:}'.format(r, j))

            occam_me, occam_te = occam_trial(seed=s, max_observations=r)
            em_me, em_te, em_ic = em_trial(seed=s, max_observations=r)

            occam_data[i, j, 0] = occam_me
            occam_data[i, j, 1] = occam_te
            em_data[i, j, 0] = em_me
            em_data[i, j, 1] = em_te
            em_data[i, j, 2] = em_ic

    np.save('occam_data', occam_data)
    np.save('em_data', em_data)

