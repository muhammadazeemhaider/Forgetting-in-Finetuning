#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from intruder_dimensions import (
    TensorStore,
    adapter_tensor_map,
    canonical_adapter_key,
    reconstruct_lora,
    vector_diagnostics,
)


class IntruderTests(unittest.TestCase):
    def test_canonical_adapter_key(self):
        key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        self.assertEqual(
            canonical_adapter_key(key),
            ("model.layers.0.self_attn.q_proj.weight", "A"),
        )
        with_default = "base_model.model.gpt_neox.layers.1.mlp.dense.lora_B.default.weight"
        self.assertEqual(
            canonical_adapter_key(with_default),
            ("gpt_neox.layers.1.mlp.dense.weight", "B"),
        )

    def test_adapter_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter_model.safetensors"
            a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            b = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
            save_file(
                {
                    "base_model.model.layer.proj.lora_A.weight": a,
                    "base_model.model.layer.proj.lora_B.weight": b,
                },
                path,
            )
            store = TensorStore(path)
            mapping = adapter_tensor_map(store)
            got = reconstruct_lora(torch.zeros(2, 2), store, mapping["layer.proj.weight"], 0.5, torch.device("cpu"))
            self.assertTrue(torch.equal(got, 0.5 * (b @ a)))

    def test_known_intruder_and_sign_invariance(self):
        dimension = 16
        u0 = torch.eye(dimension)
        s0 = torch.arange(dimension, 0, -1, dtype=torch.float32)
        vh0 = torch.eye(dimension)
        first = torch.ones(dimension) / dimension**0.5
        # Complete the deliberately diffuse first direction into an orthogonal basis.
        seed = torch.eye(dimension)
        seed[:, 0] = first
        ut, _ = torch.linalg.qr(seed)
        if torch.dot(ut[:, 0], first) < 0:
            ut[:, 0].neg_()
        st = s0.clone()
        vht = torch.eye(dimension)
        rows = vector_diagnostics(u0, s0, vh0, ut, st, vht, top_k=1, threshold=0.5)
        self.assertLess(rows[0]["left_max_abs_cosine"], 0.5)
        self.assertTrue(rows[0]["is_intruder"])
        rows_flipped = vector_diagnostics(u0, s0, vh0, -ut, st, vht, top_k=1, threshold=0.5)
        self.assertAlmostEqual(rows[0]["left_max_abs_cosine"], rows_flipped[0]["left_max_abs_cosine"])

    def test_identity_has_no_intruders(self):
        matrix = torch.randn(12, 7)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        rows = vector_diagnostics(u, s, vh, u, s, vh, top_k=7, threshold=0.5)
        self.assertFalse(any(row["is_intruder"] for row in rows))
        self.assertTrue(all(abs(row["left_max_abs_cosine"] - 1.0) < 1e-5 for row in rows))


if __name__ == "__main__":
    unittest.main()
