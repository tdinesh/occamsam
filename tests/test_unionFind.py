import unittest
import numpy as np
from itertools import combinations, chain

from utilities import UnionFind


def random_grouping(n, k):
    elements = np.arange(n)

    indptr = [0] + np.sort(np.random.choice(np.arange(1, n), size=k - 1, replace=True)).tolist() + [n]
    indices = np.random.permutation(elements)
    groups = []
    for i in range(k):
        groups.append(frozenset(np.sort(indices[indptr[i]:indptr[i + 1]]).tolist()))
    return set(groups)


def sample_groups(groups):
    pairs = []
    for g in groups:

        if len(g) < 2:
            continue

        sample_pairs = list(combinations(g, 2))
        random_sample = iter(np.random.permutation(np.arange(len(sample_pairs))))

        uncovered = g.copy()
        pair = np.random.permutation(sample_pairs[next(random_sample)]).tolist()
        uncovered = uncovered.difference(set(pair))
        pairs.append(pair)

        while len(uncovered) > 0:
            next_sample = next(random_sample)
            pair = np.random.permutation(sample_pairs[next_sample]).tolist()
            if pair[0] in uncovered and pair[1] in uncovered:
                random_sample = chain(random_sample, [next_sample])
                continue
            uncovered = uncovered.difference(set(pair))
            pairs.append(pair)

    return pairs


class TestUnionFind(unittest.TestCase):

    def test_simple(self):

        np.random.seed(0)
        groups = random_grouping(16, 3)
        elements = set(chain(*groups))
        pairs = sample_groups(groups)

        union_find = UnionFind()
        for el in elements:
            union_find.insert(el)
        for pair in pairs:
            union_find.union(pair[0], pair[1])

        uf_groups = set(frozenset(v) for v in union_find.sets().values())
        setdiff = groups.symmetric_difference(uf_groups)

        self.assertTrue(len(setdiff) == 0)


if __name__ == '__main__':
    unittest.main()
