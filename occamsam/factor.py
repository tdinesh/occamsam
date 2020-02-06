import numpy as np


class PriorFactor(object):

    def __init__(self, var, A, b, sigma):
        """
        Initializes a prior Gaussian factor on a single variable as follows
            exp^(|| A * x - b ||^2)

        :param var: Variable corresponding to x
        :param A: Linear transformation of Variable x
        :param b: Prior
        :param sigma: Noise
        """

        assert (A.shape[0] == b.size), "Measurement not in transformation codomain"
        assert (A.shape[1] == var.dim), "Variable not in transformation domain"
        assert (sigma.size == b.size), "Measurement and sigma must be the same dimension"

        self.var = var
        self.A = A
        self.b = b

        self.sigma = sigma


class LinearFactor(object):

    def __init__(self, head, tail, A1, A2, b, sigma):
        """
        Initializes a linear Gaussain factor between two variables, modeled as follows
            exp^(|| A1 * x1 - A2 * x2 - b ||^2)

        :param head: Head Variable corresponding to x1
        :param tail: Tail Variable corresponding to x2
        :param A1: Linear transformation of Variable x1
        :param A2: Linear transformation of Variable x2
        :param b: Measurement vector
        :param sigma: Measurement noise
        """

        assert (A1.shape[0] == b.size), "Measurement not in head transformation codomain"
        assert (A2.shape[0] == b.size), "Measurement not in tail transformation codomain"
        assert (A1.shape[1] == head.dim), "Head Variable not in transformation domain"
        assert (A2.shape[1] == tail.dim), "Tail Variable not in transformation domain"
        assert (sigma.size == b.size), "Measurement and sigma must be the same dimension"

        self.head = head
        self.tail = tail

        self.A1 = A1
        self.A2 = A2
        self.b = b

        self.sigma = sigma


class OdometryFactor(LinearFactor):

    def __init__(self, start, end, R, t, sigma):
        """
        Odometry factors are linear Gaussian factors between pairs of position variables modeled as follows
            exp^(|| p2 - p1 - R*t ||^2)

        Note that the rotation R transforms t from the robot frame to a shared frame of reference.
        This can be supplied using the Compass module.

        :param start: Starting PointVariable
        :param end: Ending PointVariable
        :param R: Coordinate frame to express the displacement in
        :param t: Displacement/translation vector
        :param sigma: Odometry_noise
        """

        t_ = np.dot(R, t)
        if np.isscalar(t_):
            t_ = np.array([t_])

        I = np.eye(t_.shape[0], start.dim)
        super(OdometryFactor, self).__init__(end, start, I, I, t_, sigma)


class ObservationFactor(LinearFactor):

    def __init__(self, point, landmark, R, d, sigma):
        """
        Observation factors are linear Gaussian factors between position and landmark variables
            exp^(|| m  -  R * p  - d ||^2)

        :param point: PointVariable at which the landmark is observed
        :param landmark: LandmarkVariable which is observed
        :param R: Coordinate frame of the landmark w.r.t. to the position
        :param d: Range measurement from the point position to the closest point of the landmark
        :param sigma: Observation noise
        """

        if np.isscalar(d):
            d = np.array([d])

        # TODO When |d| > |m|, this will pick up the first |m| components... is that the bx we want?
        I = np.eye(d.shape[0], landmark.dim)
        super(ObservationFactor, self).__init__(landmark, point, I, R, d, sigma)
