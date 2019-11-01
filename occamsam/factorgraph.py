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

    def __init__(self, var1, var2, A1, A2, b):

        """
        Initializes a linear Gaussain factor between two variables, modeled as follows
            exp^(|| A1 * x1 - A2 * x2 - b ||^2)

        :param var1: Head Variable corresponding to x1
        :param var2: Tail Variable corresponding to x2
        :param A1: Linear transformation of Variable x1
        :param A2: Linear transformation of Variable x2
        :param b: Measurement vector
        """

        self.var1 = var1
        self.var2 = var2

        self.A1 = A1
        self.A2 = A2
        self.b = b


class OdometryFactor(LinearFactor):

    def __init__(self, point1, point2, R, t):

        """
        Odometry factors are linear Gaussian factors between pairs of position variables modeled as follows
            exp^(|| p2 - p1 - R*t ||^2)

        Note that the rotation R transforms t from the robot frame to a shared frame of reference.
        This can be supplied using the Compass module.

        :param point1: Starting PointVariable
        :param point2: Ending PointVariable
        :param R: Coordinate frame to express the displacement in
        :param t: Displacement/translation vector
        """

        I = np.eye(t.shape[0], point1.dim)
        super(OdometryFactor, self).__init__(point2, point1, I, I, np.dot(R, t))


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

        self.graph = nx.DiGraph()
        self.margin_window = None

    def add_factor(self, factor):

        """
        Adds a LinearFactor as an edge in our factor graph

        While it may seem redundant to store the entire factor object in the edge data, this will allow us to recover which
        variables were associated by the later optimization
        """

        self.graph.add_edge(factor.var2, factor.var1, factor=factor)

    def observation_matrix(self):

        nodelist = [] + []
        edgelist = list(self.graph.edges())


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










