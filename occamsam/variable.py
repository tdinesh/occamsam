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

        :param dim: Dimensionality of the point variable
        """

        super(PointVariable, self).__init__(dim)


class LandmarkVariable(Variable):

    def __init__(self, dim, class_label, mass=1):
        """
        A landmark variable corresponds to the position of a landmark in some frame.

        Note that landmark labels must be mutually exclusive. Only landmarks with the same label will be automatically
        associated with one another.

        :param dim: Dimensionality of the position variable
        :param class_label: Equivalence class to which the landmark belongs
        :param mass: Measure of landmark size, weight, or importance
        """

        super(LandmarkVariable, self).__init__(dim)

        self.class_label = class_label
        self.mass = mass
