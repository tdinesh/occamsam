import unittest
import numpy as np
from itertools import chain

from occamsam.utilities import UnionFind, random_groups, sample_pairs


class TestUnionFind(unittest.TestCase):

    def test_simple(self):

        np.random.seed(0)
        groups = random_groups(16, 3)
        elements = set(chain(*groups))
        pairs = sample_pairs(groups)

        union_find = UnionFind()
        for el in elements:
            union_find.insert(el)
        for pair in pairs:
            union_find.union(pair[0], pair[1])

        uf_groups = set(frozenset(v) for v in union_find.set_map().values())
        set_diff = groups.symmetric_difference(uf_groups)

        self.assertTrue(len(set_diff) == 0)

    def test_big(self):

        np.random.seed(15)
        groups = random_groups(3212, 80)
        elements = set(chain(*groups))
        pairs = sample_pairs(groups)

        union_find = UnionFind()
        for el in elements:
            union_find.insert(el)
        for pair in pairs:
            union_find.union(pair[0], pair[1])

        uf_groups = set(frozenset(v) for v in union_find.set_map().values())
        set_diff = groups.symmetric_difference(uf_groups)

        self.assertTrue(len(set_diff) == 0)


if __name__ == '__main__':
    unittest.main()
