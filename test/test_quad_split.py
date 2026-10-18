# OpenSculptBoy
# Apache License, Version 2.0
"""The split of a quad along its shorter diagonal, whatever the rounding of its corners."""

import unittest

import torch

from anny.utils.mesh_utils import DIAGONAL_TIE, triangulate_faces

QUAD = [0, 1, 2, 3]
ALONG_AC = [[0, 1, 2], [2, 3, 0]]
ALONG_BD = [[0, 1, 3], [3, 1, 2]]


def eye_square(grow: float = 0.0) -> torch.Tensor:
    """A 3 mm square at the place of an eyeball quad, its diagonal a-c longer by ``grow``."""
    square = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    square[2, :2] += grow
    return 0.003 * square + torch.tensor([0.0283, -0.1391, 0.7271], dtype=torch.float64)


class TestQuadSplit(unittest.TestCase):
    def test_ties_split_alike(self):
        """A square splits along b-d after a change of any coordinate by one unit in the last place."""
        for corner in range(4):
            for axis in range(3):
                for direction in (-1.0, 1.0):
                    vertices = eye_square()
                    value = vertices[corner, axis]
                    vertices[corner, axis] = torch.nextafter(value, value + direction)
                    with self.subTest(corner=corner, axis=axis, direction=direction):
                        self.assertEqual(triangulate_faces(vertices, [QUAD]), ALONG_BD)

    def test_shorter_diagonal(self):
        """Diagonals that differ by more than the tie split along the shorter one."""
        grow = 1000 * DIAGONAL_TIE
        self.assertEqual(triangulate_faces(eye_square(-grow), [QUAD]), ALONG_AC)
        self.assertEqual(triangulate_faces(eye_square(grow), [QUAD]), ALONG_BD)


if __name__ == "__main__":
    unittest.main()
