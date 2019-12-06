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

        if max_free_points is None:
            self.max_free_points = sys.maxsize
        else:
            self.max_free_points = max_free_points

        self._free_point_buffer = OrderedDict()  # map of point variables to their associated column in A and B
        self._A = DBSRMatrix()
        self._B = DBSRMatrix()

        self._landmark_index = {}  # map of landmark variables to their associated column in A and B
        self._H = DBSRMatrix()

        self._array_index = {'d': {}, 't': {}}  # map of point variables to rows in which they participate in a factor
        self._d = []
        self._t = []

    def append(self, f):

        if isinstance(f, OdometryFactor):

            if len(self._t) == 0:
                if f.tail.position is None:
                    init_position = np.zeros(f.tail.dim)
                else:
                    init_position = f.tail.position
                self._append_to_free_buffer(f.tail)
                self._array_index['t'][f.tail] = [0]
                self._B.append_row(0, np.eye(len(init_position)))
                self._t.append(init_position)

            row = len(self._t)

            self._append_to_free_buffer(f.tail)
            self._append_to_array_index(f.tail, row, 't')

            self._append_to_free_buffer(f.head)
            self._append_to_array_index(f.head, row, 't')

            self._B.append_row([list(self._free_point_buffer.keys()).index(f.tail),
                                list(self._free_point_buffer.keys()).index(f.head)],
                               [-f.A2, f.A1])
            self._t.append(f.b)

        elif isinstance(f, ObservationFactor):

            if f.head not in self._landmark_index:
                self._landmark_index[f.head] = len(self._landmark_index)

            self._append_to_free_buffer(f.tail)
            self._append_to_array_index(f.tail, len(self._d), 'd')

            self._H.append_row(self._landmark_index[f.head], f.A1)
            self._A.append_row(list(self._free_point_buffer.keys()).index(f.tail), -f.A2)
            self._d.append(f.b)

        else:
            raise TypeError

        self._maintain_buffers()

    def _append_to_free_buffer(self, point):
        if point not in self._free_point_buffer:
            self._free_point_buffer[point] = {}

    def _append_to_array_index(self, point, row, meas_type):
        assert (meas_type == 'd' or meas_type == 't'), "Invalid meas_type"
        if point not in self._array_index[meas_type]:
            self._array_index[meas_type][point] = []
        self._array_index[meas_type][point].append(row)

    def _maintain_buffers(self):

        num_free = len(self._free_point_buffer)
        if num_free > self.max_free_points:

            # traverse buffer from back to front marginalizing points with known positions
            num_old = num_free - self.max_free_points
            for oldest_point in list(self._free_point_buffer)[:num_old]:
                if oldest_point.position is not None:
                    x = self._free_point_buffer.popitem(last=False)[0]
                    assert (oldest_point is x), "Buffer ordering corrupted"
                    self._marginalize(oldest_point, 0)

    def _marginalize(self, point, col):
        for debug_iter, (A, b, array_index) in enumerate([(self._A, self._d, self._array_index['d']),
                                                          (self._B, self._t, self._array_index['t'])]):
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

    @property
    def observation_system(self):
        _H = self._H.to_bsr()
        _A = self._A.to_bsr()
        zero_fill = sp.sparse.bsr_matrix((_H.shape[0] - _A.shape[0], _A.shape[1]))

        A = sp.sparse.bmat([[_H, sp.sparse.bmat([[zero_fill], [_A]])]]).tocsr()
        d = np.block(self._d)
        return A, d

    @property
    def odometry_system(self):
        A = self._B.to_bsr().tocsr()
        d = np.block(self._t)[-A.shape[0]:]
        return A, d
