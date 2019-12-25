import unittest
import numpy as np

import factorgraph
from equivalence import equivalence_matrix
from simulator import new_simulation


# class TestEquivalence(unittest.TestCase):
#
#     def test_identity(self):
#
#         sim = new_simulation(point_dim=3, landmark_dim=3, num_points=10, num_landmarks=30, seed=501)
#         fg = factorgraph.GaussianFactorGraph()
#         for f in sim.fix_landmarks(list(range(sim.num_landmarks))).factors():
#             fg.add_factor(f)
#
#         landmarks = sim.landmark_variables
#
#         _, equiv_pairs = equivalence_matrix(fg.landmarks)
#         equiv_pairs = [(landmarks[i], landmarks[j]) for i, j in equiv_pairs]
#
#         equiv_groups, _ = sim.equivalences()
#
#         fg._merge_landmarks(equiv_pairs)
#         unique_landmarks = fg.landmarks
#         unique_order = [sim.unique_index_map[landmarks.index(k)] for k in unique_landmarks]
#
#         self.assertEqual(len(fg.correspondence_map.set_map().keys()), sim.num_unique_landmarks)
#         fg_groups = set(frozenset(g) for g in fg.correspondence_map.set_map().values())
#         diff = equiv_groups.symmetric_difference(fg_groups)
#         self.assertTrue(len(diff) == 0)
#
#         A, b = fg.observation_system
#         x = np.concatenate((np.ravel(sim.unique_landmark_positions[unique_order, :]), np.ravel(sim.point_positions)))
#
#         self.assertEqual(b.size, A.shape[0])
#         self.assertEqual(x.size, A.shape[1])
#         self.assertTrue(np.allclose(A.dot(x), b))


if __name__ == '__main__':
    unittest.main()
