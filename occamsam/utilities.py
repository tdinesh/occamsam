import factor
import factorgraph
import simulator
import variable


def sim_to_factorgraph(sim):

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

    return fg

if __name__ == '__main__':

    sim = simulator.Simulator(3, 1, 100, 20)
    fg = sim_to_factorgraph(sim)

    fg.observation_system(5)
