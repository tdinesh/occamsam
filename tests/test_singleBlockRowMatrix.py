import unittest
import numpy as np
import scipy as sp
import scipy.sparse

from factorgraph import SingleBlockRowMatrix
from scipy.sparse import bsr_matrix


class TestSingleBlockRowMatrix(unittest.TestCase):

    def test_construction(self):

        nblocks = 10

        np.random.seed(nblocks)
        data = np.random.random((nblocks, 2, 3))
        indices = np.random.randint(0, 7, size=nblocks)
        indptr = np.arange(nblocks + 1)

        sbrm = SingleBlockRowMatrix()
        for i in range(nblocks):
            sbrm.append_row(data[i], indices[i])

        bsr = bsr_matrix((data, indices, indptr))

        self.assertTrue(np.allclose(bsr.todense(), sbrm.tobsr().todense()))

    def test_change(self):

        nblocks = 10

        np.random.seed(nblocks)
        data = np.random.random((nblocks, 2, 2))
        indices = np.random.randint(0, 5, size=nblocks)
        indptr = np.arange(nblocks + 1)

        sbrm = SingleBlockRowMatrix()
        for i in range(nblocks):
            sbrm.append_row(data[i], indices[i])

        new_block = np.random.random((2, 2))
        data[3] = new_block
        indices[3] = 1
        sbrm.set_row(new_block, 3, 1)

        bsr = bsr_matrix((data, indices, indptr))

        self.assertTrue(np.allclose(bsr.todense(), sbrm.tobsr().todense()))

    def test_remove_row(self):

        nblocks = 12

        np.random.seed(nblocks)
        data = np.random.random((nblocks, 2, 2))
        indices = np.random.randint(0, 5, size=nblocks)

        sbrm = SingleBlockRowMatrix()
        for i in range(nblocks):
            sbrm.append_row(data[i], indices[i])

        data = np.delete(data, 3, 0)
        indices = np.delete(indices, 3)
        indptr = np.arange(nblocks)

        sbrm.remove_row(3)

        bsr = bsr_matrix((data, indices, indptr))

        self.assertTrue(np.allclose(bsr.todense(), sbrm.tobsr().todense()))

    def test_remove_col(self):

        nblocks = 11

        np.random.seed(nblocks)
        data = np.random.random((nblocks, 1, 1))
        indices = np.random.randint(0, 5, size=nblocks)

        sbrm = SingleBlockRowMatrix()
        for i in range(nblocks):
            sbrm.append_row(data[i], indices[i])

        data = np.delete(data, np.flatnonzero(indices == 3), 0)
        indices = np.delete(indices, np.flatnonzero(indices == 3))
        indices[indices > 3] += -1
        indptr = np.arange(len(indices) + 1)

        sbrm.remove_col(3)

        bsr = bsr_matrix((data, indices, indptr))

        self.assertTrue(np.allclose(bsr.todense(), sbrm.tobsr().todense()))


if __name__ == '__main__':
    unittest.main()
