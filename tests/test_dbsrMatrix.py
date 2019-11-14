import unittest
import numpy as np

from sparse import DBSRMatrix
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

        self.assertTrue(np.allclose(bsr.todense(), sbrm.tobsr().todense()))

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

        self.assertTrue(np.allclose(bsr.todense(), sbrm.tobsr().todense()))

    # def test_remove_col(self):

    #     nblocks = 11

    #     np.random.seed(nblocks)
    #     data = np.random.random((nblocks, 3, 2))
    #     indices = np.random.randint(0, 5, size=nblocks)

    #     sbrm = DBSRMatrix()
    #     for i in range(nblocks):
    #         sbrm.append_row(indices[i], data[i])

    #     data = np.delete(data, np.flatnonzero(indices == 3), 0)
    #     indices = np.delete(indices, np.flatnonzero(indices == 3))
    #     indices[indices > 3] += -1
    #     indptr = np.arange(len(indices) + 1)

    #     sbrm.remove_col(3)

    #     bsr = bsr_matrix((data, indices, indptr))

    #     self.assertTrue(np.allclose(bsr.todense(), sbrm.tobsr().todense()))

    def test_merge_col(self):

        nblocks = 13

        np.random.seed(nblocks)
        data = np.random.random((nblocks, 2, 3))
        indices = np.random.randint(0, 4, size=nblocks)
        indptr = np.arange(len(indices) + 1)

        sbrm = DBSRMatrix()
        for i in range(nblocks):
            sbrm.append_row(indices[i], data[i])

        indices[indices == 0] = 2
        indices += -1

        sbrm.merge_col(0, 2)

        bsr = bsr_matrix((data, indices, indptr))

        self.assertTrue(np.allclose(bsr.todense(), sbrm.tobsr().todense()))


if __name__ == '__main__':
    unittest.main()
