import sys
import numpy as np
import scipy as sp
import scipy.sparse
from collections import OrderedDict

from factor import LinearFactor, ObservationFactor, OdometryFactor, PriorFactor
from variable import LandmarkVariable, PointVariable
from sparse import DBSRMatrix
from utilities import UnionFind


class GaussianFactorGraph(object):

    def __init__(self, free_point_window=None):

        if free_point_window is None:
            self._free_point_window = sys.maxsize
        else:
            self._free_point_window = free_point_window

        self.point_dim = None
        self._points = OrderedDict()
        self._free_point_buffer = OrderedDict()  # map of point variables to their associated column in A and B
        self._Ap = DBSRMatrix()
        self._Bp = DBSRMatrix()

        self.landmark_dim = None
        self._landmark_buffer = OrderedDict()  # map of landmark variables to their associated column in H
        self._Am = DBSRMatrix()

        self._array_index = {'d': {}, 't': {}}  # map of point variables to rows in which they participate in a factor
        self._d = []
        self._t = []
        self._sigma_d = []
        self._sigma_t = []

        self.correspondence_map = UnionFind()

    def add_factor(self, f):
        """
        Adds a LinearFactor as an edge in our factor graph

        :param f: OdometryFactor or ObservationFactor to append to the sparse system
        """
        assert (isinstance(f, (PriorFactor, LinearFactor))), "Expected Factor type, got %s" % type(f)

        if isinstance(f, OdometryFactor):

            if self.point_dim is None:
                self.point_dim = f.tail.dim
            assert (self.point_dim == f.head.dim), "Expected head of dimension %d, got %d" % (self.point_dim, f.head.dim)

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
            assert (self.point_dim == f.tail.dim), "Expected point of dimension %d, got %d" % (self.point_dim, f.tail.dim)
            if self.landmark_dim is None:
                self.landmark_dim = f.head.dim
            assert (self.landmark_dim == f.head.dim), "Expected landmark of dimension %d, got %d" % (self.landmark_dim, f.head.dim)

            self._append_to_landmark_buffer(f.head)
            self._append_to_free_buffer(f.tail)
            self._append_to_array_index(f.tail, len(self._d), 'd')

            self._Am.append_row(list(self._landmark_buffer.keys()).index(self.correspondence_map.find(f.head)), f.A1.copy())
            self._Ap.append_row(list(self._free_point_buffer.keys()).index(f.tail), -f.A2.copy())
            self._d.append(f.b.copy())
            self._sigma_d.append(f.sigma)

        elif isinstance(f, PriorFactor):

            if isinstance(f.var, PointVariable):
                if self.point_dim is None:
                    self.point_dim = f.var.dim
                assert (self.point_dim == f.var.dim), "Expected point of dimension %d, got %d" % (self.point_dim, f.var.dim)
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
        Appends the point variable point to the buffer of free points, which are all the point variables whose positions
        are yet unknown and will be optimized over

        :param point: point variable
        """

        assert isinstance(point, PointVariable), "Expected type PointVariable, got %s instead" % type(point)
        self._free_point_buffer[point] = None
        self._points[point] = None

    def _append_to_landmark_buffer(self, landmark):
        """
        Appends the landmark variable point to a buffer, where a landmarks position in the buffer corresponds to its index
        in the H matrix

        :param landmark: landmark variable
        """

        assert isinstance(landmark, LandmarkVariable), "Expected type LandmarkVariable, got %s instead" % type(landmark)
        self._landmark_buffer[landmark] = None
        self.correspondence_map.insert(landmark)

    def _append_to_array_index(self, point, row, meas_type):
        """
        Appends row to the list of row factors in which the point variable participates in.

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
        Maintains the free point buffer by evicting old point variables if the number of elements exceeds max_free_points

        Evicted point variables are marginalized out of both measurement matrices using their last known position

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
        Marginalizes the point variable point in column col out of both the odometry and observation system

        (A, -A') * (p, p')^T = b ==> A * p = b - (-A' * p')

        :param point: point variable to marginalize
        :param col: column index corresponding to the point variable
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

        Am = self._Am.to_bsr()
        Ap = self._Ap.to_bsr()

        n_missing_cols = (Ap.blocksize[1] * len(self._free_point_buffer)) - Ap.shape[1]
        col_padding = sp.sparse.bsr_matrix((Ap.shape[0], n_missing_cols))

        n_missing_rows = Am.shape[0] - Ap.shape[0]
        row_padding = sp.sparse.bsr_matrix((n_missing_rows, Ap.shape[1]))

        Am = Am.tocsr()
        Ap = sp.sparse.bmat([[row_padding, None], [Ap, col_padding]]).tocsr()
        d = np.block(self._d)

        return Am, Ap, d

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

        Bp = self._Bp.to_bsr().tocsr()
        t = np.block(self._t)[-Bp.shape[0]:]
        return Bp, t

    @property
    def points(self):
        return list(self._points.keys())

    @property
    def landmarks(self):
        return list(self._landmark_buffer.keys())


