import unittest
import numpy as np
import itertools

from occamsam.sparse import DBSRMatrix
from scipy.sparse import bsr_matrix


class TestSingleBlock(unittest.TestCase):

    def test_construction(self):

        nblocks = 10

        np.random.seed(nblocks)
        data = np.random.random((nblocks, 2, 3))
        indices = np.random.randint(0, 7, size=nblocks)
        indptr = np.arange(nblocks + 1)

        sbrm = DBSRMatrix()
        for i in range(nblocks):
            sbrm.append_row(indices[i], data[i])

        bsr = bsr_matrix((data, indices, indptr))

        self.assertTrue(np.allclose(bsr.todense(), sbrm.to_bsr().todense()))

    # def test_insert_block(self):

    #     nblocks = 10

    #     np.random.seed(nblocks)
    #     data = np.random.random((nblocks, 2, 3))
    #     indices = np.random.randint(0, 7, size=nblocks)
    #     indptr = np.arange(nblocks + 1)

    #     sbrm = DBSRMatrix()
    #     for i in range(nblocks):
    #         sbrm.append_row(indices[i], data[i])

    #     bsr = bsr_matrix((data, indices, indptr))

    #     self.assertTrue(np.allclose(bsr.todense(), sbrm.to_bsr().todense()))

    def test_remove_row(self):

        nblocks = 12

        np.random.seed(nblocks)
        data = np.random.random((nblocks, 2, 2))
        indices = np.random.randint(0, 5, size=nblocks)

        sbrm = DBSRMatrix()
        for i in range(nblocks):
            sbrm.append_row(indices[i], data[i])

        data = np.delete(data, 3, 0)
        indices = np.delete(indices, 3)
        indptr = np.arange(nblocks)

        sbrm.remove_row(3)

        bsr = bsr_matrix((data, indices, indptr))

        self.assertTrue(np.allclose(bsr.todense(), sbrm.to_bsr().todense()))

    def test_remove_col(self):

        nblocks = 11

        np.random.seed(nblocks)
        data = np.random.random((nblocks, 3, 2))
        indices = np.random.randint(0, 5, size=nblocks)

        sbrm = DBSRMatrix()
        for i in range(nblocks):
            sbrm.append_row(indices[i], data[i])

        data = np.delete(data, np.flatnonzero(indices == 3), 0)
        indices = np.delete(indices, np.flatnonzero(indices == 3))
        indices[indices > 3] += -1
        indptr = np.arange(len(indices) + 1)

        sbrm.remove_col(3)

        bsr = bsr_matrix((data, indices, indptr))

        self.assertTrue(np.allclose(bsr.todense(), sbrm.to_bsr().todense()))

    def test_merge_cols(self):

        nblocks = 13

        np.random.seed(nblocks)
        data = np.random.random((nblocks, 2, 3))
        indices = np.random.randint(0, 6, size=nblocks)
        indptr = np.arange(len(indices) + 1)

        sbrm = DBSRMatrix()
        for i in range(nblocks):
            sbrm.append_row(indices[i], data[i])

        indices[indices == 2] = 1
        indices[indices > 2] += -1

        sbrm.merge_cols((1, 2))

        bsr = bsr_matrix((data, indices, indptr))

        self.assertTrue(np.allclose(bsr.todense(), sbrm.to_bsr().todense()))


class TestMultiBlock(unittest.TestCase):

    def test_construction(self):

        nblocks = 25
        nrows = 11

        np.random.seed(nblocks)

        data = np.random.random((nblocks, 2, 3))
        indices = np.random.randint(0, 7, size=nblocks)
        indptr = np.array(
            [0] + np.sort(np.random.choice(np.arange(1, nblocks - 1), size=nrows - 1, replace=False)).tolist() + [
                nblocks])

        sbrm = DBSRMatrix()
        for i in range(nrows):
            sbrm.append_row(indices[indptr[i]:indptr[i + 1]], data[indptr[i]:indptr[i + 1]])

        bsr = bsr_matrix((data, indices, indptr))

        self.assertTrue(np.allclose(bsr.todense(), sbrm.to_bsr().todense()))

    def test_construction_list(self):

        nblocks = 25
        nrows = 11

        np.random.seed(nblocks)

        data = np.random.random((nblocks, 2, 3))
        indices = np.random.randint(0, 7, size=nblocks)
        indptr = np.array(
            [0] + np.sort(np.random.choice(np.arange(1, nblocks - 1), size=nrows - 1, replace=False)).tolist() + [
                nblocks])

        sbrm = DBSRMatrix()
        for i in range(nrows):
            sbrm.append_row(indices[indptr[i]:indptr[i + 1]].tolist(), data[indptr[i]:indptr[i + 1]].tolist())

        bsr = bsr_matrix((data, indices, indptr))

        self.assertTrue(np.allclose(bsr.todense(), sbrm.to_bsr().todense()))

    def test_construction_column_repeat(self):

        nblocks = 50
        nrows = 2

        np.random.seed(nblocks)

        data = np.random.random((nblocks, 2, 3))
        indices = np.random.randint(0, 7, size=nblocks)
        indptr = np.array(
            [0] + np.sort(np.random.choice(np.arange(1, nblocks - 1), size=nrows - 1, replace=False)).tolist() + [
                nblocks])

        sbrm = DBSRMatrix()
        for i in range(nrows):
            sbrm.append_row(indices[indptr[i]:indptr[i + 1]], data[indptr[i]:indptr[i + 1]])

        bsr = bsr_matrix((data, indices, indptr))

        self.assertTrue(np.allclose(bsr.todense(), sbrm.to_bsr().todense()))

    def test_merge(self):

        n_blocks = 100
        n_rows = 5
        n_cols = 10
        col_dim = 3

        np.random.seed(n_blocks)
        data = np.random.random((n_blocks, 2, col_dim))
        indices = np.random.randint(0, n_cols, size=n_blocks)
        indptr = np.array(
            [0] + np.sort(np.random.choice(np.arange(1, n_blocks - 1), size=n_rows - 1, replace=False)).tolist() + [
                n_blocks])

        sbrm = DBSRMatrix()
        for i in range(n_rows):
            sbrm.append_row(indices[indptr[i]:indptr[i + 1]], data[indptr[i]:indptr[i + 1]])

        np_mat = bsr_matrix((data, indices, indptr)).todense()

        pair = np.random.choice(np.arange(n_cols), size=2, replace=False)
        sbrm.merge_cols(tuple(pair.tolist()))

        dest = min(pair)
        src = max(pair)
        np_mat[:, col_dim * dest:col_dim * dest + col_dim] += np_mat[:, col_dim * src:col_dim * src + col_dim]
        np_mat = np.delete(np_mat, np.arange(col_dim * src, col_dim * src + col_dim), 1)

        self.assertTrue(np.allclose(np_mat, sbrm.to_bsr().todense()))



if __name__ == '__main__':
    unittest.main()
