import sys
from collections import OrderedDict

import numpy as np
import scipy as sp
from scipy import sparse

from factor import ObservationFactor, OdometryFactor
from sparse import DBSRMatrix


class DynamicMeasurementSystem(object):

    def __init__(self, max_free_points=None):
        """
        :param max_free_points:
        """

        self._Ap = DBSRMatrix()
        self._free_point_buffer = OrderedDict()  # map of point variables to their associated column in Ap
        if max_free_points is None:
            self.max_free_points = sys.maxsize
        else:
            self.max_free_points = max_free_points

        self._d = []
        self._marginalization_index = {}  # map of point variables to rows in which they participate in a factor

    def append(self, f):
        pass

    def to_sparse(self):
        pass

    def _append_to_free_buffer(self, point):
        self._free_point_buffer[point] = {}
        num_free = len(self._free_point_buffer)
        if num_free > self.max_free_points:

            # traverse buffer from back to front marginalizing points with known positions
            for oldest_index, oldest_point in reversed(list(enumerate(self._free_point_buffer))[self.max_free_points:]):
                if oldest_point.position is not None:
                    x = self._free_point_buffer.popitem(last=True)
                    assert (oldest_point is x), "Buffer ordering corrupted"
                    self._marginalize(oldest_point, oldest_index)

    def _marginalize(self, point, col):
        col_data = self._Ap.get_col(col)
        n_blocks = len(col_data)
        n_rows = len(self._marginalization_index[point])
        assert (n_rows == n_blocks), "Expected %d blocks in column, got %d instead" % (n_rows, n_blocks)
        for i, (row, block) in reversed(list(enumerate(col_data))):
            self._d[self._marginalization_index[point][i]] += np.dot(block, point.position)
            self._Ap.remove_row(row)


class ObservationSystem(DynamicMeasurementSystem):

    def __init__(self, max_free_points=None):
        """
        :param max_free_points:
        """
        super(ObservationSystem, self).__init__(max_free_points=max_free_points)
        self._Am = DBSRMatrix()
        self._landmark_index = {}  # map of landmark variables to their associated column in Am

    def append(self, f):

        assert (isinstance(f, ObservationFactor)), "Expected type ObservationFactor, got %s" % type(f)

        if f.head not in self._landmark_index:
            self._landmark_index[f.head] = len(self._landmark_index)

        if f.tail not in self._free_point_buffer:
            self._append_to_free_buffer(f.tail)

        if f.tail not in self._marginalization_index:
            self._marginalization_index[f.tail] = []

        self._Am.append_row(self._landmark_index[f.head], f.A1)
        self._Ap.append_row(list(self._free_point_buffer.keys()).index(f.tail), f.A2)
        self._d.append(f.b)
        self._marginalization_index[f.tail].append(len(self._d))

    def to_sparse(self):
        A = sp.sparse.bmat([[None, None], [self._Am.to_bsr(), self._Ap.to_bsr()]])
        d = np.block(self._d)
        return A, d


class OdometrySystem(DynamicMeasurementSystem):

    def __init__(self, init_position, max_free_points=None):
        """
        :param init_position:
        :param max_free_points:
        """
        super(OdometrySystem, self).__init__(max_free_points)
        self._Ap.append_row(0, np.eye(len(init_position)))
        self._d.append(init_position)

    def append(self, f):

        assert (isinstance(f, OdometryFactor)), "Expected type OdometryFactor, got %s" % type(f)

        if f.head not in self._free_point_buffer:
            self._free_point_buffer[f.head] = {}

        if f.tail not in self._free_point_buffer:
            self._append_to_free_buffer(f.tail)

        if f.tail not in self._marginalization_index:
            self._marginalization_index[f.tail] = []

        self._Ap.append_row(list(self._free_point_buffer.keys()).index(f.tail), f.A2)
        self._d.append(f.b)
        self._marginalization_index[f.tail].append(len(self._d))

    def to_sparse(self):
        A = self._Ap.to_bsr()
        d = np.block(self._d)
        return A, d
