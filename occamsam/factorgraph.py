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

        super(LandmarkVariable, self).__init__(dim )

        self.eqclass = label


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

        # if not isinstance(d, np.ndarray):
        #     d = np.array(d)

        # if not d.shape:
        #     d = d[None]

        I = np.eye(d.shape[0], landmark.dim)
        super(ObservationFactor, self).__init__(landmark, point, I, H, d)

class GaussianFactorGraph(object):

    def __init__(self):

        self.graph = nx.DiGraph()

    def add_factor(self, factor):

        """
        Adds a LinearFactor as an edge in our factor graph

        While it may seem redundant to store the entire factor object in the edge data, this will allow us to recover which
        variables were associated by the later optimization
        """

        self.graph.add_edge(factor.tail, factor.head, factor=factor)

    def observation_system(self, point_window=None):

        observations = [(u, v, f) for (u, v, f) in self.graph.edges.data('factor') if isinstance(f, ObservationFactor)]
        landmarks = [node for node in self.graph.nodes() if isinstance(node, LandmarkVariable)]
        points = [node for node in self.graph.nodes() if isinstance(node, PointVariable)]

        if point_window is None:
            point_window = len(points)
        fixed_points = len(points) - point_window

        rows = np.sum([f.b.size for (u, v, f) in observations])
        lmcols = np.sum([lm.dim for lm in landmarks])
        ptcols = np.sum([pt.dim for pt in points[-point_window:]])
        fixcols = int(np.sum([pt.dim for pt in points[:fixed_points]]))

        Am = sp.sparse.lil_matrix((rows, lmcols))
        Ap = sp.sparse.lil_matrix((rows, ptcols))
        Af = sp.sparse.lil_matrix((rows, fixcols))

        landmark_index = dict([(landmark, landmark.dim * i) for i, landmark in enumerate(landmarks)])
        point_index = dict([(point, point.dim * i) for i, point in enumerate(points[-point_window:])])
        fixed_index = dict([(point, point.dim * i) for i, point in enumerate(points[:fixed_points])])

        ei = 0
        for (u, v, f) in observations:

            k = f.b.size

            vi = landmark_index[v]
            Am[ei:ei+k, vi:vi+v.dim] = f.A1

            if u in point_index.keys():

                ui = point_index[u]
                Ap[ei:ei+k, ui:ui+u.dim] = -f.A2

            else:

                ui = fixed_index[u]
                Af[ei:ei+k, ui:ui+u.dim] = -f.A2

            ei += k

        params = {'Am': Am.asformat('csr'), 'Ap': Ap.asformat('csr')}

        return params


    def observation_array(self):
        return np.concatenate([np.array(f.b) for (u, v, f) in self.graph.edges.data('factor') if isinstance(f, ObservationFactor)])

    def odometry_array(self):
        return np.concatenate([np.array(f.b) for (u, v, f) in self.graph.edges.data('factor') if isinstance(f, OdometryFactor)])

    def draw(self):

        """
        TODO: Replace with a hook to Cytoscape of Graphiz as recommended
        """

        import matplotlib.pyplot as plt

        plt.plot()
        nx.draw(self.graph)
        plt.show()










