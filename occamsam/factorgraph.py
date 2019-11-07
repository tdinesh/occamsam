import numpy as np
import networkx as nx
import scipy as sp

import scipy.sparse

from factor import OdometryFactor, ObservationFactor
from variable import PointVariable, LandmarkVariable


class GaussianFactorGraph(object):

    def __init__(self):

        # For python versions < 3.6, we need to use Ordered(Graph) to access nodes in the order they are presented
        # In python 3.6+, dicts are ordered by default
        self._graph = nx.OrderedDiGraph()

    def add_factor(self, factor):
        """
        Adds a LinearFactor as an edge in our factor graph

        While it may seem redundant to store the entire factor object in the edge data, this will allow us to recover which
        variables were associated by the later optimization
        """

        self._graph.add_edge(factor.tail, factor.head, factor=factor)

        self.variables = self._graph.nodes()
        self.factors = self._graph.edges()

    def observation_system(self, num_free=None):
        """
        Returns the linear system of observation constraints on the landmark and pose variables
            Am * m - Ap * p - d = 0
            A * (m, p)^T - d = 0

        If a num_free is specified, we marginalize over the first [1, ... , |p| - num_free] point variables,
            Am * m - Ap' * p' - (Af * p'' + d') = 0
            Am * m - Ap' * p' - d = 0
            A * (m, p')^T - d = 0

        If num_free is larger than |p|, all points will be free

        If less than (|p| - num_free) point variables have known positions, all those that do will be marginalized over

        :param num_free: Number of point variables to include as free parameters
        :return: A: Set of linear observation constraints
        :return: d: Array of distance measurements
        """

        observations = [(u, v, f) for (u, v, f) in self._graph.edges.data('factor') if isinstance(f, ObservationFactor)]
        landmarks = [node for node in self._graph.nodes() if isinstance(node, LandmarkVariable)]
        points = [node for node in self._graph.nodes() if isinstance(node, PointVariable)]

        if num_free is None:
            num_fixed = 0
            num_free = len(points)
        else:
            num_updated = len([p for p in points if p.position is not None])
            num_fixed = min(max(0, len(points) - num_free), num_updated)
            num_free = len(points) - num_fixed

        free_points = points[-num_free:] if num_free else []
        fixed_points = points[:num_fixed]

        rows = np.sum([f.b.size for (u, v, f) in observations])
        landmark_cols = np.sum([lm.dim for lm in landmarks])
        free_cols = int(np.sum([pt.dim for pt in free_points]))
        fix_cols = int(np.sum([pt.dim for pt in fixed_points]))

        Am = sp.sparse.lil_matrix((rows, landmark_cols))
        Ap = sp.sparse.lil_matrix((rows, free_cols))
        Af = sp.sparse.lil_matrix((rows, fix_cols))
        d = np.zeros(rows)

        landmark_index = dict([(landmark, landmark.dim * i) for i, landmark in enumerate(landmarks)])
        free_index = dict([(point, point.dim * i) for i, point in enumerate(free_points)])
        fixed_index = dict([(point, point.dim * i) for i, point in enumerate(fixed_points)])

        ei = 0
        for (u, v, f) in observations:

            k = f.b.size

            vi = landmark_index[v]
            Am[ei:ei + k, vi:vi + v.dim] = f.A1
            d[ei:ei + k] = f.b

            if u in free_index.keys():
                ui = free_index[u]
                Ap[ei:ei + k, ui:ui + u.dim] = f.A2
            else:
                ui = fixed_index[u]
                Af[ei:ei + k, ui:ui + u.dim] = f.A2

            ei += k

        if num_fixed > 0:
            Af = Af.asformat('csr')
            p = np.concatenate([np.array(p.position) for p in fixed_points])
            d = Af.dot(p) + d

        A = sp.sparse.hstack([Am, -Ap], format='csr')

        return A, d

    def odometry_system(self, num_free=None):
        """
        Returns the linear system of odometry constraints on the pose variables
            Ap * p - t = 0

        If a num_free is specified, we marginalize over the first [1, ... , |p| - num_free] point variables,
            Ap' * p' - (Af * p'' + t') = 0
            A * p' - t = 0

        If num_free is larger than |p|, all points will be free

        If less than (|p| - num_free) point variables have known positions, all those that do will be marginalized over

        :param num_free: Number of point variables to include as free parameters
        :return: A: Set of linear odometry constraints
        :return: t: Array of translation measurements
        """
        observations = [(u, v, f) for (u, v, f) in self._graph.edges.data('factor') if isinstance(f, OdometryFactor)]
        points = [node for node in self._graph.nodes() if isinstance(node, PointVariable)]

        if num_free is None:
            num_fixed = 0
            num_free = len(points)
        else:
            num_updated = len([p for p in points if p.position is not None])
            num_fixed = min(max(0, len(points) - num_free), num_updated)
            num_free = len(points) - num_fixed

        free_points = points[-num_free:] if num_free else []
        fixed_points = points[:num_fixed]

        rows = np.sum([f.b.size for (u, v, f) in observations])
        free_cols = int(np.sum([pt.dim for pt in free_points]))
        fix_cols = int(np.sum([pt.dim for pt in fixed_points]))

        Ap = sp.sparse.lil_matrix((rows, free_cols))
        Af = sp.sparse.lil_matrix((rows, fix_cols))
        t = np.zeros(rows)

        free_index = dict([(point, point.dim * i) for i, point in enumerate(free_points)])
        fixed_index = dict([(point, point.dim * i) for i, point in enumerate(fixed_points)])

        ei = 0
        for (u, v, f) in observations:

            k = f.b.size

            t[ei:ei+k] = f.b

            if v in free_index.keys():
                vi = free_index[v]
                Ap[ei:ei + k, vi:vi + v.dim] = f.A1
            else:
                vi = fixed_index[v]
                Af[ei:ei + k, vi:vi + v.dim] = -f.A1

            if u in free_index.keys():
                ui = free_index[u]
                Ap[ei:ei + k, ui:ui + u.dim] = -f.A2
            else:
                ui = fixed_index[u]
                Af[ei:ei + k, ui:ui + u.dim] = f.A2

            ei += k

        if num_fixed > 0:
            Af = Af.asformat('csr')
            p = np.concatenate([np.array(p.position) for p in fixed_points])
            t = Af.dot(p) + t

        A = Ap.asformat('csr')

        return A, t

    # def contract_variables(self, u, v):

    def draw(self):

        """
        TODO: Replace with a hook to Cytoscape of Graphiz as recommended
        """

        import matplotlib.pyplot as plt

        plt.plot()
        nx.draw(self._graph)
        plt.show()
