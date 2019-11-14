from factor import LinearFactor, ObservationFactor, OdometryFactor
from sparse import DBSRMatrix
from variable import LandmarkVariable, PointVariable

import numpy as np
import scipy as sp
import scipy.sparse

import networkx as nx


class GaussianFactorGraph(object):

    def __init__(self, free_point_window=None):

        # For python versions < 3.6, we need to use Ordered(Graph) to access nodes in the order they are presented
        # In python 3.6+, dicts are ordered by default
        self._graph = nx.OrderedDiGraph()

        self.free_point_window = free_point_window  # number of point variables to include as free parameters in
        # the linear systems

        # publicly access
        self.variables = self._graph.nodes()
        self.factors = self._graph.edges()

    def add_factor(self, f):
        """
        Adds a LinearFactor as an edge in our factor graph

        """

        assert (isinstance(f, LinearFactor)), "Expected type LinearFactor, got %s" % type(f)

        self._graph.add_edge(f.tail, f.head, factor=f)

    # def contract_variables(self, u, v):

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

        observations = [(u, v, f) for (u, v, f) in self._graph.edges.data('factor') if isinstance(f, ObservationFactor)]
        landmarks = [node for node in self._graph.nodes() if isinstance(node, LandmarkVariable)]
        points = [node for node in self._graph.nodes() if isinstance(node, PointVariable)]

        if self.free_point_window is None:
            num_fixed = 0
            num_free = len(points)
        else:
            num_updated = len([p for p in points if p.position is not None])
            num_fixed = min(max(0, len(points) - self.free_point_window), num_updated)
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

        observations = [(u, v, f) for (u, v, f) in self._graph.edges.data('factor') if isinstance(f, OdometryFactor)]
        points = [node for node in self._graph.nodes() if isinstance(node, PointVariable)]

        if self.free_point_window is None:
            num_fixed = 0
            num_free = len(points)
        else:
            num_updated = len([p for p in points if p.position is not None])
            num_fixed = min(max(0, len(points) - self.free_point_window), num_updated)
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

            t[ei:ei + k] = f.b

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

    def draw(self):

        """
        TODO: Replace with a hook to Cytoscape of Graphiz as recommended
        """

        import matplotlib.pyplot as plt

        plt.plot()
        nx.draw(self._graph)
        plt.show()


if __name__ == '__main__':

    mat = DBSRMatrix()

    np.random.seed(11)
    mat._indices = np.random.randint(0, 200, size=10000).tolist()

    for _ in range(500):
        i, j = np.random.choice(mat._indices, size=2, replace=False)
        i, j = np.sort([i, j])

        mat.merge_columns1(i, j)
        mat.merge_columns2(i, j)
