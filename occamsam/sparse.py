import numpy as np
import scipy as sp
import scipy.sparse
import itertools


class DBSRMatrix(object):

    def __init__(self):
        self._indptr = [0]
        self._indices = []
        self._data = []

    def append_row(self, cols, blocks):

        if isinstance(cols, (int, np.integer)):
            cols = np.array([cols])
            blocks = np.array([blocks])
        elif isinstance(cols, list):
            cols = np.array(cols)
            blocks = np.array(blocks)

        assert isinstance(cols, np.ndarray), "Expected type %s for cols, got %s" % (list, type(cols))
        assert isinstance(blocks, np.ndarray), "Expected type %s for blocks, got %s" % (list, type(blocks))
        assert (len(cols) == len(blocks)), "cols and blocks must contain the same number of elements"

        order = np.argsort(cols)
        cols = cols[order]
        blocks = blocks[order]

        self._indptr.append(self._indptr[-1])
        for col, group in itertools.groupby(zip(*(cols, blocks)), key=lambda x: x[0]):
            self._indptr[-1] += 1
            self._indices.append(col)
            self._data.append(np.zeros_like(blocks[0]))
            for _, block in group:
                self._data[-1] += block

    def insert_block(self, row, col, block):

        if row > len(self._indptr) - 1:
            self.append_row(col, block)
            return

        row_cols = self._indices[self._indptr[row]:self._indptr[row + 1]]
        if col in row_cols:
            col_index = row_cols.index(col)
            self._data[col_index] += block
        else:
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

    def merge_cols(self, pair):

        assert isinstance(pair, tuple), "Expected tuple for pair, got %s" % type(pair)
        assert len(pair) == 2, "pair must be of length 2, got %d" % len(pair)

        dest = min(pair)
        src = max(pair)

        if src == dest:
            return

        indices = np.array(self._indices)
        indices[indices == src] = dest
        indices[indices > src] -= 1
        self._indices = indices.tolist()

    def copy_col(self, src, dest):
        assert isinstance(dest, (int, np.int)), "Expected integer for dest, got %s" % type(dest)
        assert isinstance(src, (int, np.int)), "Expected integer for src, got %s" % type(src)

        if src == dest:
            return

        self._indices = [dest if x == src else x for x in self._indices]

    def to_bsr(self):

        if len(self._data) == 0:
            return sp.sparse.bsr_matrix([])

        data = np.array(self._data)
        indices = np.array(self._indices)
        indptr = np.array(self._indptr)
        return sp.sparse.bsr_matrix((data, indices, indptr))
