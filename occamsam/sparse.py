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
            # TODO test this path
            assert (len(cols) == len(blocks)), "columns and blocks must contain the same number of elements"
            self._indptr.append(self._indptr[-1] + len(blocks))
            self._indices.extend(cols)
            self._data.extend(blocks)

    def insert_block(self, row, col, block):

        if row > len(self._indptr) - 1:
            self.append_row(col, block)
            return

        # TODO check if col already present in row
        end_index = self._indptr[row + 1]
        self._data.insert(end_index, block)
        self._indices.insert(end_index, col)

        indptr = np.array(self._indptr)
        indptr[row + 1:] += 1
        self._indptr = indptr.tolist()

    def remove_row(self, row):

        start_index = self._indptr[row]
        stop_index = self._indptr[row + 1]

        del self._data[start_index:stop_index]
        del self._indices[start_index:stop_index]

        n = stop_index - start_index
        self._indptr[row + 1:] = [x - n for x in self._indptr[row + 1:]]
        del self._indptr[row + 1]

    def remove_col(self, col):

        n_rows = len(self._indptr) - 1
        for i in reversed(range(n_rows)):

            start_index = self._indptr[i]
            stop_index = self._indptr[i + 1]

            if col not in self._indices[start_index:stop_index]:
                continue

            col_index = self._indices[start_index:stop_index].index(col)
            del self._indices[start_index + col_index]
            del self._data[start_index + col_index]

            self._indptr[i + 1:] = [x - 1 for x in self._indptr[i + 1:]]
            if self._indptr[i] == self._indptr[i + 1]:
                del self._indptr[i + 1]

        self._indices = [x - 1 if x > col else x for x in self._indices]

    def get_col(self, col):

        col_data = []

        n_rows = len(self._indptr) - 1
        for i in range(n_rows):

            start_index = self._indptr[i]
            stop_index = self._indptr[i + 1]

            if col not in self._indices[start_index:stop_index]:
                continue

            col_index = self._indices[start_index:stop_index].index(col)
            col_data.append((i, self._data[start_index + col_index]))

        return col_data

    def merge_col(self, src, dest):
        # TODO make this accept a list

        indices = np.array(self._indices)
        indices[indices == src] = dest
        indices[indices > src] -= 1
        self._indices = indices.tolist()

    def to_bsr(self):

        if len(self._data) == 0:
            return sp.sparse.bsr_matrix([])

        data = np.array(self._data)
        indices = np.array(self._indices)
        indptr = np.arange(len(self._indices) + 1)
        return sp.sparse.bsr_matrix((data, indices, indptr))
