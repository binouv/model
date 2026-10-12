"""CPU-only integrity and mechanism tests: no Qwen weights and no science scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

src=Path(__file__).resolve().parents[1]/"src"
sys.path.insert(0,str(src))
from dataset import (build, load, save, DATASET_COUNT, SEED, canonical_sha)
from latent_reasoner import LatentReasoner, two_digit_loss
from train import train_one, restore_model, evaluate


class H1cSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dataset_frozen_and_disjoint(self):
        items=build()
        self.assertEqual(len(items),1700)
        self.assertEqual(len({r["group"] for r in items}),1700)
        self.assertEqual(sum(r["split"]=="train" for r in items),1200)
        self.assertEqual(sum(r["split"]=="val" for r in items),200)
        self.assertEqual(sum(r["split"]=="test" for r in items),300)
        self.assertEqual(canonical_sha(items),"ac764eed8a0e06486dee8452219293941e208618fd95cbfa7495395f71184af7")

    def test_oracle_validity(self):
        from h1_dataset import verify_python
        for row in build()[::41]:
            a,b,c=row["operands"]
            self.assertEqual(verify_python(row["operator"],a,b,c),tuple(row["digits"]))
            self.assertNotIn("Answer: "+row["answer"],row["question"])

    def test_roundtrip_and_hash_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            manifest=save(out)
            rows,found=load(out)
            self.assertEqual(len(rows),DATASET_COUNT)
            self.assertEqual(found["dataset_sha256_canonical"],manifest["dataset_sha256_canonical"])
            path=out/"h1c_dataset.jsonl"
            path.write_bytes(path.read_bytes()+b"\n")
            with self.assertRaises(RuntimeError):load(out)

    def test_model_shapes(self):
        x=torch.randn(3,32)
        model=LatentReasoner(32,16)
        for steps in (0,2,8):
            logits=model(x,steps)
            self.assertEqual(tuple(logits.shape),(3,2,10))
            self.assertTrue(torch.isfinite(logits).all())

    def test_fixed_parameter_count_for_steps(self):
        model=LatentReasoner(32,16)
        n=sum(p.numel() for p in model.parameters())
        for steps in (0,2,8):
            _=model(torch.randn(2,32),steps)
            self.assertEqual(n,sum(p.numel() for p in model.parameters()))

    def test_gradient_through_recurrent_core(self):
        torch.manual_seed(6)
        model=LatentReasoner(32,16)
        x=torch.randn(8,32)
        truth=torch.randint(0,10,(8,2))
        logits=model(x,8)
        loss=two_digit_loss(logits,truth)
        loss.backward()
        self.assertIsNotNone(model.recurrent.weight_ih.grad)
        self.assertGreater(float(model.recurrent.weight_ih.grad.abs().sum()),0)

    def test_causal_trajectory_zero_matches_direct(self):
        model=LatentReasoner(32,16)
        x=torch.randn(5,32)
        _,traj=model(x,8,save_trajectory=True)
        torch.testing.assert_close(traj[:,0],model(x,0))
        torch.testing.assert_close(traj[:,8],model(x,8))
        self.assertFalse(torch.allclose(traj[:,0],traj[:,8]))

    def test_adaptive_always_full_at_high_threshold(self):
        model=LatentReasoner(32,16)
        x=torch.randn(10,32)
        a,steps=model.adaptive(x,.999999999)
        self.assertTrue((steps==8).all())
        torch.testing.assert_close(a,model(x,8))

    def test_adaptive_outputs_and_confidence_bounds(self):
        model=LatentReasoner(32,16)
        x=torch.randn(10,32)
        final,steps=model.adaptive(x,.65)
        self.assertEqual(tuple(final.shape),(10,2,10))
        self.assertTrue(set(steps.tolist())<={0,2,4,8})
        c=model.confidence(final)
        self.assertTrue(((c>=0)&(c<=1)).all())

    def test_invalid_shapes_and_early_exit(self):
        model=LatentReasoner(32,16)
        with self.assertRaises(ValueError):model(torch.randn(2,33))
        with self.assertRaises(ValueError):model(torch.randn(2,32),-1)
        with self.assertRaises(ValueError):model.adaptive(torch.randn(2,32),1.0)
        with self.assertRaises(ValueError):two_digit_loss(torch.randn(2,20),torch.zeros(2,2,dtype=torch.long))

    def test_train_checkpoint_restores(self):
        torch.manual_seed(11)
        x=torch.randn(24,32)
        y=torch.randint(0,10,(24,2))
        split={"train":list(range(16)),"val":list(range(16,20)),"test":list(range(20,24))}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            model,rec=train_one("fixture",2,11,x,y,split,out,steps=3,batch=6,lr=.0015,device="cpu")
            restored=restore_model(out,rec)
            a=model(x,2)
            b=restored(x,2)
            torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-7)
            self.assertTrue((out/rec["checkpoint"]["path"]).is_file())
            self.assertEqual(evaluate(model,x,y,2,split["val"])["n"],4)

if __name__=="__main__":unittest.main(verbosity=2)