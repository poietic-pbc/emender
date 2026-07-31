#!/usr/bin/env python3
"""Dependency-free simulation for exact hierarchical DiLoCo averaging math."""

import unittest


def global_average(rows):
    width = len(rows[0])
    return [sum(row[i] for row in rows) / len(rows) for i in range(width)]


def hierarchical_weighted_average(rows, group_size):
    partials = []
    counts = []
    for start in range(0, len(rows), group_size):
        group = rows[start:start + group_size]
        counts.append(len(group))
        partials.append([sum(row[i] for row in group) for i in range(len(rows[0]))])
    total = sum(counts)
    return [sum(partial[i] for partial in partials) / total for i in range(len(rows[0]))]


def naive_average_of_averages(rows, group_size):
    group_means = []
    for start in range(0, len(rows), group_size):
        group = rows[start:start + group_size]
        group_means.append([sum(row[i] for row in group) / len(group) for i in range(len(rows[0]))])
    return [sum(row[i] for row in group_means) / len(group_means) for i in range(len(rows[0]))]


class HierarchicalDilocoMathTest(unittest.TestCase):
    def test_weighted_hierarchy_matches_global_average_for_unequal_groups(self):
        rows = [
            [1.0, 2.0, 3.0],
            [5.0, 8.0, 13.0],
            [100.0, 200.0, 300.0],
        ]
        self.assertEqual(hierarchical_weighted_average(rows, group_size=2), global_average(rows))
        self.assertNotEqual(naive_average_of_averages(rows, group_size=2), global_average(rows))

    def test_equal_groups_make_average_of_averages_accidentally_correct(self):
        rows = [
            [1.0, 2.0],
            [3.0, 4.0],
            [10.0, 20.0],
            [30.0, 40.0],
        ]
        self.assertEqual(hierarchical_weighted_average(rows, group_size=2), global_average(rows))
        self.assertEqual(naive_average_of_averages(rows, group_size=2), global_average(rows))


if __name__ == '__main__':
    unittest.main()
