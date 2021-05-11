import sys
import numpy as np
import scipy as sp
import scipy.sparse
from collections import OrderedDict

from occamsam.factor import LinearFactor, ObservationFactor, OdometryFactor, PriorFactor
from occamsam.variable import LandmarkVariable, PointVariable
from occamsam.sparse import DBSRMatrix
from occamsam.utilities import UnionFind


class GaussianFactorGraph(object):

    def __init__(self, free_point_window=None):
        """
        A class for dynamically maintaining a set of linear motion and observation measurements with Gaussian noise
            collected over the course of robot navigation

        While the name of the class suggests graph-like functionality, which is to be added in the future, its primary
            purpose is to maintain a set of linear systems constraining robot and observation positions

        By defining a free_point_window, the class will automatically only maintain measurement factors containing
            at least one of the last free_point_window number of robot point positions as free variables in the
            linear systems. Those points which eventually become too old are automatically marginalized using their
            last known position

        :param free_point_window: An integer declaring the number of most recent robot point positions, which are
            included as free variables in the linear system(s)
        """

        if free_point_window is None:
            self._free_point_window = sys.maxsize
        else:
            self._free_point_window = free_point_window

        self.point_dim = None  # Dimensionality of the point variables, which is detected and then enforced
        #   following the first point variable encounter
        self._points = OrderedDict()  # Collection of all point variables encountered
        self._free_point_buffer = OrderedDict()  # Collection of the last free_point_window point variables. The
        #   index of each PointVariable in the buffer corresponds to its
        #   column index within _Ap and _Bp.
        #   if free_point_window is None, _free_point_buffer == _points.

        self.landmark_dim = None  # Dimensionality of the landmark variables, detected and then enforced following the
        #   first encounter
        self._landmark_buffer = OrderedDict()  # Collection of all the unique landmark variables encountered. The index
        #   of each LandmarkVariable in the buffer corresponds to its column
        #   index within _Am

        self._Ap = DBSRMatrix()  # Observation matrix transforming robot point positions
        self._Am = DBSRMatrix()  # Observation matrix transforming landmark positions

        self._Bp = DBSRMatrix()  # Odometry matrix relating robot point positions

        self._array_index = {'d': {}, 't': {}}  # Map of variables to the rows in which they participate in a factor
        self._d = []  # List of range measurements from a robot point to a landmark
        self._t = []  # List of robot translations
        self._sigma_d = []  # List of estimated standard deviations for each range measurement
        self._sigma_t = []  # List of estimated standard deviations for each translation measurement

        self.correspondence_map = UnionFind()  # Map of landmarks to their parent landmark. Landmarks sharing the same
        #   parent form their own group. Associated landmarks share a single
        #   column within the linear systems

    def add_factor(self, f):
        """
        Adds a new measurement factor to the system.

        If the length of the _free_point_buffer equals the number free_point_window and the factor f contains a new
            PointVariable, the oldest PointVariable in the _free_point_buffer is evicted and marginalized out of both
            the Observation and Odometry linear systems using its last known position

        :param f: PriorFactor, OdometryFactor, or ObservationFactor to be added
        """

        assert (isinstance(f, (PriorFactor, LinearFactor))), "Expected Factor type, got %s" % type(f)

        if isinstance(f, OdometryFactor):

            if self.point_dim is None:
                self.point_dim = f.tail.dim
            assert (self.point_dim == f.head.dim), "Expected head of dimension %d, got %d" % (
                self.point_dim, f.head.dim)

            row = len(self._t)
            self._append_to_free_buffer(f.tail)
            self._append_to_array_index(f.tail, row, 't')
            self._append_to_free_buffer(f.head)
            self._append_to_array_index(f.head, row, 't')

            self._Bp.append_row([list(self._free_point_buffer.keys()).index(f.tail),
                                 list(self._free_point_buffer.keys()).index(f.head)],
                                [-f.A2.copy(), f.A1.copy()])
            self._t.append(f.b.copy())
            self._sigma_t.append(f.sigma)

        elif isinstance(f, ObservationFactor):

            if self.point_dim is None:
                self.point_dim = f.tail.dim
            assert (self.point_dim == f.tail.dim), "Expected point of dimension %d, got %d" % (
                self.point_dim, f.tail.dim)
            if self.landmark_dim is None:
                self.landmark_dim = f.head.dim
            assert (self.landmark_dim == f.head.dim), "Expected landmark of dimension %d, got %d" % (
                self.landmark_dim, f.head.dim)

            self._append_to_landmark_buffer(f.head)
            self._append_to_free_buffer(f.tail)
            self._append_to_array_index(f.tail, len(self._d), 'd')

            self._Am.append_row(list(self._landmark_buffer.keys()).index(self.correspondence_map.find(f.head)),
                                f.A1.copy())
            self._Ap.append_row(list(self._free_point_buffer.keys()).index(f.tail), -f.A2.copy())
            self._d.append(f.b.copy())
            self._sigma_d.append(f.sigma)

        elif isinstance(f, PriorFactor):

            if isinstance(f.var, PointVariable):
                if self.point_dim is None:
                    self.point_dim = f.var.dim
                assert (self.point_dim == f.var.dim), "Expected point of dimension %d, got %d" % (
                    self.point_dim, f.var.dim)
                self._append_to_free_buffer(f.var)
                self._append_to_array_index(f.var, len(self._t), 't')
                self._Bp.append_row(list(self._free_point_buffer.keys()).index(f.var), f.A)
                self._t.append(f.b.copy())
                self._sigma_t.append(f.sigma)
            else:
                raise TypeError

        else:
            raise TypeError

        self._maintain_buffers()

    def _append_to_free_buffer(self, point):
        """
        Appends the PointVariable point to the buffer of free points and points

        :param point: PointVariable
        """

        assert isinstance(point, PointVariable), "Expected type PointVariable, got %s instead" % type(point)
        self._free_point_buffer[point] = None
        self._points[point] = None

    def _append_to_landmark_buffer(self, landmark):
        """
        Appends the LandmarkVariable landmark to the landmark buffer

        The LandmarkVariable is also added the correspondence map as its own parent and group

        :param landmark: LandmarkVariable
        """

        assert isinstance(landmark, LandmarkVariable), "Expected type LandmarkVariable, got %s instead" % type(landmark)
        self.correspondence_map.insert(landmark)
        self._landmark_buffer[self.correspondence_map.find(landmark)] = None

    def _append_to_array_index(self, point, row, meas_type):
        """
        Appends point to the list of row factors in which the point variable participates in.

        Separate lists are maintained for translation and distance measurements

        :param point: point variable
        :param row: row of the linear system where the column corresponding to point has a non-zero value
        :param meas_type: 'd' for distance measurement or 't' for translation measurement
        """

        assert (meas_type == 'd' or meas_type == 't'), "Invalid meas_type"
        if point not in self._array_index[meas_type]:
            self._array_index[meas_type][point] = []
        self._array_index[meas_type][point].append(row)

    def _maintain_buffers(self):
        """
        Maintains the _free_point_buffer by evicting old PointVariables if the number of elements exceeds
            the number specified by free_point_window

        Evicted PointVariables are marginalized out of both measurement matrices using their last known position

        If the variable's position is still None, the point is kept in the free buffer until it is set
        """

        num_free = len(self._free_point_buffer)
        if num_free > self._free_point_window:

            # traverse buffer from back to front marginalizing points with known positions
            num_old = num_free - self._free_point_window
            for oldest_point in list(self._free_point_buffer)[:num_old]:
                if oldest_point.position is not None:
                    x = self._free_point_buffer.popitem(last=False)[0]
                    assert (oldest_point is x), "Buffer ordering corrupted"
                    self._marginalize(oldest_point, 0)

    def _marginalize(self, point, col):
        """
        Marginalizes the PointVariable point located in column col out of both _Ap and _Bp

            (A, -A') * (p, p')^T = b ==> A * p = b - (-A' * p')

        :param point: PointVariable to marginalize
        :param col: column index corresponding to point in _Ap and _Bp
        """

        for debug_iter, (A, b, array_index) in enumerate([(self._Ap, self._d, self._array_index['d']),
                                                          (self._Bp, self._t, self._array_index['t'])]):
            col_data = A.get_col(col)
            n_blocks = len(col_data)
            if n_blocks > 0:
                n_rows = len(array_index[point])
                assert (n_rows == n_blocks), "Expected %d blocks in column, got %d instead" % (n_rows, n_blocks)
                for i, (row, block) in reversed(list(enumerate(col_data))):
                    b[array_index[point][i]] -= np.dot(block, point.position)
                    A.insert_block(row, col, np.zeros_like(block))
                    del array_index[point][i]
            A.remove_col(col)

    def merge_landmarks(self, pairs):
        """
        Merges every pair of landmarks (LandmarkVariable u, LandmarkVariable v) in the list pairs

        Groups of LandmarkVariables all sharing the same parent will share the same column within _Am after each call

        Only the parent LandmarkVariable for each group remains in the _landmark_buffer after each call

        :param pairs:
        """

        for u, v in pairs:
            self.correspondence_map.union(u, v)

        set_map = self.correspondence_map.set_map()
        for super_landmark in set_map:
            landmark_index_map = dict((k, i) for i, k in enumerate(self._landmark_buffer.keys()))
            correspondence_set = set(set_map[super_landmark])
            unmerged_landmarks = correspondence_set.intersection(self._landmark_buffer.keys()).difference(
                {super_landmark})
            super_landmark_index = landmark_index_map[super_landmark]
            unmerged_landmark_index = sorted([landmark_index_map[x] for x in unmerged_landmarks])
            for i in unmerged_landmark_index:
                self._Am.copy_col(i, super_landmark_index)
            for i in reversed(unmerged_landmark_index):
                self._Am.remove_col(i)
            for landmark in unmerged_landmarks:
                self._landmark_buffer.pop(landmark, None)

    def observation_system(self):
        """
        Returns the linear system of observation constraints between LandmarkVariables and PointVariables
            Am * m - Ap * p - d = 0
            A * (m, p)^T - d = 0

        If a free_point_window is specified in the class, the first [1, ... , |p| - free_point_window] PointVariables
            encountered are marginalized over:

            Am * m - Ap' * p' - (Af * p'' + d') = 0
            Am * m - Ap' * p' - d = 0
            A * (m, p')^T - d = 0

        If the free_point_window is larger than |p|, all PointVariables will have a corresponding column
            within the linear system

        If less than (|p| - free_point_window) PointVariables have known positions, all those that do will
            be marginalized over

        :return: Am: Matrix transforming landmark positions to the constraint space
        :return: Ap: Matrix transforming robot point positions to the constraint space
        :return: d: Array of corresponding distance measurements for each row
        :return: sigma_d: Array of corresponding noise estimates for each row
        """

        Am = self._Am.to_bsr()
        Ap = self._Ap.to_bsr()

        n_missing_cols = (Ap.blocksize[1] * len(self._free_point_buffer)) - Ap.shape[1]
        col_padding = sp.sparse.bsr_matrix((Ap.shape[0], n_missing_cols))

        n_missing_rows = Am.shape[0] - Ap.shape[0]
        row_padding = sp.sparse.bsr_matrix((n_missing_rows, Ap.shape[1]))

        Am = Am.tocsr()
        Ap = sp.sparse.bmat([[row_padding, None], [Ap, col_padding]]).tocsr()
        d = np.block(self._d if len(self._d) > 0 else np.array([]))
        sigma_d = np.block(self._sigma_d if len(self._sigma_d) > 0 else np.array([]))

        return Am, Ap, d, sigma_d

    def odometry_system(self):
        """
        Returns the linear system of translational odometric constraints between consecutive PointVariables
            Ap * p - t = 0

        If a free_point_window is specified in the class, the first [1, ... , |p| - free_point_window] PointVariables
            encountered are marginalized over:

            Ap' * p' - (Af * p'' + t') = 0
            A * p' - t = 0

        If the free_point_window is larger than |p|, all PointVariables will have a corresponding column
            within the linear system

        If less than (|p| - free_point_window) PointVariables have known positions, all those that do will
            be marginalized over

        :return: Bp: Matrix whose rows constrain the position of consecutive PointVariables
        :return: t: Array of corresponding translation measurements for each row
        :return: sigma_d: Array of corresponding noise estimates for each row
        """

        Bp = self._Bp.to_bsr().tocsr()
        t = np.block(self._t if len(self._t) > 0 else np.array([]))[-Bp.shape[0]:]
        sigma_t = np.block(self._sigma_t if len(self._sigma_t) > 0 else np.array([]))[-Bp.shape[0]:]

        return Bp, t, sigma_t

    @property
    def points(self):
        return list(self._points.keys())

    @property
    def free_points(self):
        return list(self._free_point_buffer.keys())

    @property
    def landmarks(self):
        return list(self._landmark_buffer.keys())
