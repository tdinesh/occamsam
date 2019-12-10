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

    def sets(self):
        set_map = {}
        for k, v in self._parent.items():
            set_map.setdefault(v, list()).append(k)
        return set_map
