import numpy as np
import networkx as nx
import scipy as sp

import scipy.sparse


class Variable(object):

    def __init__(self, dim):
        """
        Defines a variable to be optimized in the factor graph based on their pairwise positional relationships
        defined by different Factors.

        :param dim:  Dimensionality of the variable
        """

        self.position = None
        self.dim = dim


class PointVariable(Variable):

    def __init__(self, dim):
        """
        A position variable corresponds to a robots position within some frame.

        :param dim: Dimensionality of the position variable
        """

        super(PointVariable, self).__init__(dim)


class LandmarkVariable(Variable):

    def __init__(self, dim, label):
        """
        A landmark variable corresponds to the position of a landmark in some frame. 
        
        Note that landmark labels must be mutually exclusive. Only landmarks with the same label will be automatically
        associated with one another.

        :param dim: Dimensionality of the position variable
        :param label: Equivalence class to which the landmark belongs
        """

        super(LandmarkVariable, self).__init__(dim)

        self.eqclass = label


class PriorFactor(object):

    def __init__(self, var, A, b):
        """
        Initializes a prior Gaussian factor on a single variable as follows
            exp^(|| A * x - b ||^2)

        :param var: Variable corresponding to x
        :param A: Linear transformation of Variable x
        :param b: Prior
        """

        assert (A.shape[0] == b.size), "Measurement not in transformation codomain"
        assert (A.shape[1] == var.dim), "Variable not in transformation domain"

        self.var = var
        self.A = A
        self.b = b


class LinearFactor(object):

    def __init__(self, head, tail, A1, A2, b):
        """
        Initializes a linear Gaussain factor between two variables, modeled as follows
            exp^(|| A1 * x1 - A2 * x2 - b ||^2)

        :param head: Head Variable corresponding to x1
        :param tail: Tail Variable corresponding to x2
        :param A1: Linear transformation of Variable x1
        :param A2: Linear transformation of Variable x2
        :param b: Measurement vector
        """

        assert (A1.shape[0] == b.size), "Measurement not in head transformation codomain"
        assert (A2.shape[0] == b.size), "Measurement not in tail transformation codomain"
        assert (A1.shape[1] == head.dim), "Head Variable not in transformation domain"
        assert (A2.shape[1] == tail.dim), "Tail Variable not in transformation domain"

        self.head = head
        self.tail = tail

        self.A1 = A1
        self.A2 = A2
        self.b = b


class OdometryFactor(LinearFactor):

    def __init__(self, start, end, R, t):
        """
        Odometry factors are linear Gaussian factors between pairs of position variables modeled as follows
            exp^(|| p2 - p1 - R*t ||^2)

        Note that the rotation R transforms t from the robot frame to a shared frame of reference.
        This can be supplied using the Compass module.

        :param start: Starting PointVariable
        :param end: Ending PointVariable
        :param R: Coordinate frame to express the displacement in
        :param t: Displacement/translation vector
        """

        t_ = np.dot(R, t)

        I = np.eye(t_.shape[0], start.dim)
        super(OdometryFactor, self).__init__(end, start, I, I, t_)


class ObservationFactor(LinearFactor):

    def __init__(self, point, landmark, H, d):
        """
        Observation factors are linear Gaussian factors between position and landmark variables
            exp^(|| m  -  H * p  - d ||^2)

        Note that the transformation H can be provided by using a compass module in tandem with a feature extractor.

        :param point: PointVariable at which the landmark is observed
        :param landmark: LandmarkVariable which is observed
        :param H: Coordinate frame of the landmark w.r.t. to the position
        :param d: Distance between the position and the closest point of the landmark
        """

        I = np.eye(d.shape[0], landmark.dim)
        super(ObservationFactor, self).__init__(landmark, point, I, H, d)


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

    def draw(self):

        """
        TODO: Replace with a hook to Cytoscape of Graphiz as recommended
        """

        import matplotlib.pyplot as plt

        plt.plot()
        nx.draw(self._graph)
        plt.show()
