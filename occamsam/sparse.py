import numpy as np
import scipy as sp
import scipy.sparse


class DBSRMatrix(object):

    def __init__(self):
        self._indptr = [0]
        self._indices = []
        self._data = []

    def append_row(self, cols, blocks):

        if isinstance(cols, (int, np.integer)):
            self._indptr.append(self._indptr[-1] + 1)
            self._indices.append(cols)
            self._data.append(blocks)
        elif isinstance(cols, (list, np.ndarray)):
            assert (len(cols) == len(blocks)), "columns and blocks must contain the same number of elements"
            self._indptr.append(self._indptr[-1] + len(blocks))
            self._indices.extend(cols)
            self._data.extend(blocks)

    def insert_block(self, row, col, block):

        if row > len(self._indptr) - 1:
            self.append_row(col, block)
            return

        end_index = self._indptr[row + 1]
        self._data.insert(end_index, block)
        self._indices.insert(end_index, col)

        indptr = np.array(self._indptr)
        indptr[row + 1:] += 1
        self._indptr = indptr.tolist()

    def remove_row(self, row):

        del self._data[self._indptr[row]:self._indptr[row + 1]]
        del self._indices[self._indptr[row]:self._indptr[row + 1]]

        n = self._indptr[row + 1] - self._indptr[row]
        indptr = np.array(self._indptr)
        indptr[row + 1:] -= n
        self._indptr = indptr.tolist()
        del self._indptr[row]

    def merge_col(self, src, dest):
        indices = np.array(self._indices)
        indices[indices == src] = dest
        indices[indices > src] -= 1
        self._indices = indices.tolist()

    def tobsr(self):
        data = np.array(self._data)
        indices = np.array(self._indices)
        indptr = np.arange(len(self._indices) + 1)
        return sp.sparse.bsr_matrix((data, indices, indptr))
