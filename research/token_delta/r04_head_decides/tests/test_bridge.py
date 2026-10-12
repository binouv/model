"""Hardware-free unit tests for R04 causal alignment and no-style interface."""
import json
import sys
from pathlib import Path
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from latent_bridge import LatentBridge, assemble_writer_embeddings, answer_cross_entropy


class R04BridgeTests(unittest.TestCase):
    def test_model_lock_uses_two_immutable_shas(self):
        lock = json.loads((ROOT / "MODEL_LOCK.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["head"]["text_hidden_size"], 2560)
        self.assertEqual(lock["writer"]["text_hidden_size"], 1024)
        for role in ("head", "writer"):
            self.assertEqual(len(lock[role]["revision"]), 40)
            int(lock[role]["revision"], 16)
            self.assertTrue(lock[role]["frozen"])
        self.assertTrue(lock["bridge"]["no_writer_prompt_or_style_signal"])

    def test_shapes_and_no_style_channel(self):
        bridge = LatentBridge(12, 7, 3, 16)
        out = bridge(torch.randn(2, 12))
        self.assertEqual(tuple(out.shape), (2, 3, 7))
        self.assertEqual(bridge(torch.zeros(2, 12)).shape, out.shape)
        with self.assertRaises(ValueError):
            bridge(torch.randn(2, 9))
        # Bridge.forward accepts only one tensor. No output-length/style parameter.
        self.assertEqual(bridge.forward.__code__.co_argcount, 2)

    def test_causal_alignment_known_positions(self):
        # 3 soft prefix tokens, 2 fixed header tokens, 2 target tokens.
        # Target #0 must be read from logit position 4, target #1 from 5.
        logits = torch.zeros(1, 7, 9, requires_grad=True)
        target = torch.tensor([[3, 7]])
        loss = answer_cross_entropy(logits, target, plan_tokens=3, start_tokens=2)
        self.assertAlmostEqual(float(loss.detach()), 2.197224577, places=5)
        loss.backward()
        self.assertEqual(logits.grad[0, 4].abs().sum().item() > 0, True)
        self.assertEqual(logits.grad[0, 5].abs().sum().item() > 0, True)
        self.assertEqual(float(logits.grad[0, :4].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[0, 6].abs().sum()), 0.0)

    def test_only_bridge_receives_gradient(self):
        torch.manual_seed(20261012)
        emb = nn.Embedding(19, 7)
        output = nn.Linear(7, 19)
        emb.requires_grad_(False)
        output.requires_grad_(False)
        bridge = LatentBridge(12, 7, 3, 16)
        states = torch.randn(2, 12)
        starts = torch.tensor([[2, 4], [2, 4]])
        targets = torch.tensor([[3, 5], [6, 7]])
        x = assemble_writer_embeddings(bridge(states), starts, targets, emb)
        logits = output(x.cumsum(dim=1))  # Toy causal recurrence propagates prefix information.
        loss = answer_cross_entropy(logits, targets, 3, 2)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in emb.parameters()))
        self.assertTrue(all(p.grad is None for p in output.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in bridge.parameters()))

    def test_no_question_in_writer_embeddings(self):
        bridge = LatentBridge(12, 7, 3, 16)
        emb = nn.Embedding(19, 7)
        h = torch.randn(2, 12)
        start = torch.tensor([[1,2],[1,2]])
        answer = torch.tensor([[3,4],[5,6]])
        x = assemble_writer_embeddings(bridge(h), start, answer, emb)
        self.assertEqual(tuple(x.shape), (2, 7, 7))
        # No original question embeddings provided anywhere in the interface.


if __name__ == "__main__":
    unittest.main(verbosity=2)