import factor
import variable
from factor import LinearFactor, ObservationFactor, OdometryFactor
from variable import LandmarkVariable, PointVariable
from systems import DynamicMeasurementSystem

import networkx as nx


class GaussianFactorGraph(object):

    def __init__(self, free_point_window=None):

        # For python versions < 3.6, we need to use Ordered(Graph) to access nodes in the order they are presented
        # In python 3.6+, dicts are ordered by default
        self._graph = nx.OrderedDiGraph()

        self._measurement_system = DynamicMeasurementSystem(max_free_points=free_point_window)

        # publicly access
        self.variables = self._graph.nodes()
        self.factors = self._graph.edges()

    def add_factor(self, f):
        """
        Adds a LinearFactor as an edge in our factor graph

        """
        assert (isinstance(f, LinearFactor)), "Expected type LinearFactor, got %s" % type(f)
        self._graph.add_edge(f.tail, f.head, factor=f)

        self._measurement_system.append(f)

    # def contract_variables(self, u, v):

    @property
    def observation_system(self):
        """
        Returns the linear system of observation constraints on the landmark and pose variables
            Am * m - Ap * p - d = 0
            A * (m, p)^T - d = 0

        If a free_point_window is specified in the class, we marginalize over the first [1, ... , |p| - free_point_window]
        point variables,
            Am * m - Ap' * p' - (Af * p'' + d') = 0
            Am * m - Ap' * p' - d = 0
            A * (m, p')^T - d = 0

        If the free_point_window is larger than |p|, all points will be free

        If less than (|p| - free_point_window) point variables have known positions, all those that do will be marginalized over

        :return: A: Set of linear observation constraints
        :return: d: Array of distance measurements
        """
        return self._measurement_system.observation_system

    @property
    def odometry_system(self):
        """
        Returns the linear system of odometry constraints on the pose variables
            Ap * p - t = 0

        If a free_point_window is specified in the class, we marginalize over the first [1, ... , |p| - free_point_window]
        point variables,
            Ap' * p' - (Af * p'' + t') = 0
            A * p' - t = 0

        If the free_point_window is larger than |p|, all points will be free

        If less than (|p| - free_point_window) point variables have known positions, all those that do will be marginalized over

        :return: A: Set of linear odometry constraints
        :return: t: Array of translation measurements
        """
        return self._measurement_system.odometry_system

    @property
    def free_point_window(self):
        return self._measurement_system.max_free_points

    @free_point_window.setter
    def free_point_window(self, value):
        self._measurement_system.max_free_points = value

    def draw(self):

        """
        TODO: Replace with a hook to Cytoscape of Graphiz as recommended
        """

        import matplotlib.pyplot as plt

        plt.plot()
        nx.draw(self._graph)
        plt.show()

    def insert_simulation_factors(self, sim, fixed_points=[0]):

        point_variables = [variable.PointVariable(sim.point_dim) for _ in range(sim.num_points)]
        landmark_variables = [variable.LandmarkVariable(sim.landmark_dim, sim.landmark_labels[i])
                              for i in range(sim.num_landmarks)]

        odometry_factors = [factor.OdometryFactor(point_variables[u], point_variables[v], R, t)
                            for (u, v), R, t in zip(*sim.odometry_factors())]
        observation_factors = [factor.ObservationFactor(point_variables[u], landmark_variables[v], H, d)
                               for (u, v), H, d in zip(*sim.observation_factors())]

        for index in fixed_points:
            point_variables[index].position = sim.points[index, :]

        i = 0
        j = 0
        for pv in point_variables:

            if pv == odometry_factors[i].head:
                self.add_factor(odometry_factors[i])
                i += 1

            while j < len(observation_factors) and pv == observation_factors[j].tail:
                self.add_factor(observation_factors[j])
                j += 1
