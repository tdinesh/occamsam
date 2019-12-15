from itertools import combinations, chain

import numpy as np


class UnionFind(object):

    def __init__(self):
        self._parent = {}
        self._rank = {}

    def insert(self, node):
        if node not in self._parent:
            self._parent[node] = node
            self._rank[node] = 0

    def find(self, node):
        if self._parent[node] is not node:
            self._parent[node] = self.find(self._parent[node])
        return self._parent[node]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root is y_root:
            return

        if self._rank[x_root] < self._rank[y_root]:
            x_root, y_root = y_root, x_root

        self._parent[y_root] = x_root
        if self._rank[x_root] is self._rank[y_root]:
            self._rank[x_root] += 1

    def set_map(self):
        d = {}
        for k in self._parent:
            d.setdefault(self.find(k), list()).append(k)
        return d

    def root_map(self):
        return self._parent.copy()


def random_groups(n, k):
    elements = np.arange(n)

    indptr = [0] + np.sort(np.random.choice(np.arange(1, n), size=k - 1, replace=False)).tolist() + [n]
    indices = np.random.permutation(elements)
    groups = []
    for i in range(k):
        groups.append(frozenset(np.sort(indices[indptr[i]:indptr[i + 1]]).tolist()))
    return set(groups)


def sample_pairs(groups):
    pairs = []
    for g in groups:

        if len(g) < 2:
            continue

        all_pairs = list(combinations(g, 2))
        random_sample = iter(np.random.permutation(np.arange(len(all_pairs))))

        uncovered = g.copy()
        pair = np.random.permutation(all_pairs[next(random_sample)]).tolist()
        uncovered = uncovered.difference(set(pair))
        pairs.append(pair)

        while len(uncovered) > 0:
            next_sample = next(random_sample)
            pair = np.random.permutation(all_pairs[next_sample]).tolist()
            if pair[0] in uncovered and pair[1] in uncovered:
                random_sample = chain(random_sample, [next_sample])
                continue
            uncovered = uncovered.difference(set(pair))
            pairs.append(pair)

    shuffle_order = np.random.permutation(len(pairs)).tolist()
    pairs = [pairs[i] for i in shuffle_order]

    return pairs
