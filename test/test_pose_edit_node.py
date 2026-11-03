# OpenSculptBoy
# Apache License, Version 2.0
"""
The pose editor's parts that need no browser, in node (``viewer/test/pose_edit.mjs``): a starter's
light scan compressed with EXT_meshopt_compression reads back into the card frame, the BVH's rays
and closest points match a brute force, the points carry the screens' numbers and plain names,
and the editor's keys sit in the action registry without a conflict or a key the browser keeps.
It needs the viewer's node packages (``cd viewer && npm ci``).
"""

import pathlib
import subprocess
import unittest

from test.test_viewer_card_file import node_available

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "viewer" / "test" / "pose_edit.mjs"


@unittest.skipUnless(
    node_available() and (REPO / "viewer" / "node_modules" / "three").exists(),
    "node 22 or later and the viewer's node packages are needed",
)
class TestPoseEditNode(unittest.TestCase):
    def test_scan_bvh_points_and_keys(self):
        out = subprocess.run(
            ["node", str(SCRIPT)], cwd=REPO / "viewer", capture_output=True, text=True
        )
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        print(out.stdout.strip())


if __name__ == "__main__":
    unittest.main()
