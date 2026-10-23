# OpenSculptBoy
# Apache License, Version 2.0
"""
Checks of ``opensculptboy.texture.levels``: a normal map baked on level 2 of the detail layer and
re-encoded for level 1 decodes on level 1 to the normals that level 2 shows, the map of the
baked level passes unchanged, and the GLB file of a coarser level carries the re-encoded map.
"""

import io
import json
import pathlib
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from opensculptboy import Character, export_glb
from opensculptboy.detail import level, save_offsets
from opensculptboy.export.detail import resolve
from opensculptboy.export.document import png_bytes
from opensculptboy.texture import levels as L
from test.gltf_reader import GLB

SIZE = 1024


def _angles(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(1), -1.0, 1.0)))


def _decode(image, frames):
    rows, cols, f = frames
    local = image[rows, cols].astype(np.float64) / 127.5 - 1.0
    return np.einsum("nk,nkd->nd", local, f)


class TestLevelNormals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        folder = pathlib.Path(cls.tmp.name)
        names = {s: f"card.detail{s}.safetensors" for s in (1, 2)}
        card = Character(
            name="levels",
            detail={
                "levels": [{"subdiv": s, "offsets": n} for s, n in names.items()],
                "textures": {"subdiv": 2},
            },
            material={"normal": "card.normal.png"},
        )
        cls.folder, cls.card = folder, card
        cls.model = card.build_model(dtype=torch.float64)
        with torch.no_grad():
            cls.base = cls.model(**card.model_kwargs())["rest_vertices"][0].numpy()
        for subdiv, name in names.items():
            fine = np.asarray(level("anny", subdiv).operators[0].apply(cls.base))
            for op in level("anny", subdiv).operators[1:]:
                fine = np.asarray(op.apply(fine))
            offsets = np.zeros_like(fine)
            if subdiv == 2:  # a ripple of 1 mm across the body, which level 1 lacks
                offsets[:, 0] = 0.001 * np.sin(2 * np.pi * fine[:, 2] / 0.03)
            save_offsets(folder / name, offsets, subdiv)
        cls.fine = resolve(cls.model, card, 2, folder)
        cls.coarse = resolve(cls.model, card, 1, folder)
        mesh = cls.fine.topology
        pick = np.ones(len(mesh.triangles), bool)
        # a field of normals on level 2: its own normals tilted by a smooth turn over the atlas
        cls.fine_frames = L._frames(
            cls.fine.vertices(cls.base), mesh, np.asarray(mesh.uvs), pick, (SIZE, SIZE)
        )
        rows, cols, f = cls.fine_frames
        turn = np.stack(
            [np.sin(12.0 * cols / SIZE), np.cos(9.0 * rows / SIZE), np.ones(len(rows))],
            1,
        )
        local = turn * [0.4, 0.4, 1.0]
        local /= np.linalg.norm(local, axis=1, keepdims=True)
        image = np.zeros((SIZE, SIZE, 3), np.uint8)
        image[..., 2] = 255
        image[rows, cols] = np.round((local * 0.5 + 0.5) * 255).astype(np.uint8)
        cls.image = image
        cls.world = np.zeros((SIZE, SIZE, 3))
        cls.world[rows, cols] = _decode(image, cls.fine_frames)
        (folder / "card.normal.png").write_bytes(png_bytes(image))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_coarser_level_decodes_to_the_baked_normals(self):
        # measured: level 1 reads the level-2 map 3.2 degrees off on average (p95 8.7), and
        # the re-encoded map 0.17 degrees off (p95 0.28, the rounding to 8 bits)
        images = {"normal": png_bytes(self.image)}
        mesh = self.coarse.topology
        out = L.level_normals(
            self.model,
            self.card,
            images,
            self.coarse,
            mesh,
            None,
            self.base,
            self.folder,
        )
        moved = np.asarray(Image.open(io.BytesIO(out["normal"])).convert("RGB"))
        pick = np.ones(len(mesh.triangles), bool)
        frames = L._frames(
            self.coarse.vertices(self.base),
            mesh,
            np.asarray(mesh.uvs),
            pick,
            (SIZE, SIZE),
        )
        rows, cols, _ = frames
        known = np.zeros((SIZE, SIZE), bool)
        known[self.fine_frames[0], self.fine_frames[1]] = True
        take = known[rows, cols]
        world = self.world[rows[take], cols[take]]
        sel = (rows[take], cols[take], frames[2][take])
        before = _angles(_decode(self.image, sel), world)
        after = _angles(_decode(moved, sel), world)
        self.assertGreater(before.mean(), 2.0)
        self.assertLess(after.mean(), 0.4)
        self.assertLess(np.percentile(after, 95), 0.6)

    def test_the_baked_level_passes_unchanged(self):
        images = {"normal": png_bytes(self.image)}
        out = L.level_normals(
            self.model,
            self.card,
            images,
            self.fine,
            self.fine.topology,
            None,
            self.base,
            self.folder,
        )
        self.assertIs(out, images)

    def test_glb_of_level_one_carries_the_reencoded_map(self):
        path = self.folder / "card.json"
        path.write_text(json.dumps(self.card.to_dict()))
        out = self.folder / "level1.glb"
        export_glb(out, self.card, self.model, detail_level=1, detail_dir=self.folder)
        glb = GLB(out)
        normal = glb.json["materials"][0]["normalTexture"]["index"]
        data = glb.image_bytes(glb.json["textures"][normal]["source"])
        written = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        self.assertFalse(np.array_equal(written, self.image))


if __name__ == "__main__":
    unittest.main()
