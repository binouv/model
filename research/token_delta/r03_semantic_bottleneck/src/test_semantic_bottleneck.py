"""R03 preregistered protocol and checkpoint replay checks."""
import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

import torch
from probe_and_decode import (
  R02_ROOT,SOURCE_SHA,read_features,representation_geometry,
  FactProbe,SemanticsFirstStudent,probe_measure,evaluate_joint
)
from real_qwen_latent import split_facts, question, response

class TestR03(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_pretrained_cache_is_exact(self):
        path=R02_ROOT/"qwen_prompt_features.pt"
        self.assertTrue(path.is_file())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),SOURCE_SHA)
        x,train,test,texts=read_features(path)
        self.assertEqual(x.shape,(264,1024))
        self.assertEqual((len(train),len(test)),(168,48))
        self.assertEqual(len(texts),264)
        self.assertFalse(set(train)&set(test))
        self.assertTrue(torch.isfinite(x).all())
        geom=representation_geometry(x[:168])
        self.assertTrue(0<=geom["mean_centered_l2"])

    def test_probe_shape(self):
        x=torch.randn(4,1024)
        for name in ("linear","mlp"):
            p=FactProbe(name)
            self.assertEqual(p(x).shape,(4,3,6))
            measures=probe_measure(p,x,split_facts()[0][:4])
            self.assertEqual(measures["n"],4)

    def test_student_style_independent_latent(self):
        net=SemanticsFirstStudent()
        f=torch.randn(3,1024)
        logits,z=net.features(f)
        self.assertEqual(tuple(logits.shape),(3,3,6))
        self.assertEqual(tuple(z.shape),(3,48))
        self.assertEqual(net.renderer.generate(z,torch.zeros(3,dtype=torch.long)).shape,(3,17))

    def test_output_and_checkpoint_replay(self):
        out=Path(os.environ.get("TOKEN_DELTA_R03_RESULTS","../results"))
        if not (out/"metrics.json").is_file():self.skipTest("No completed R03 run")
        m=json.loads((out/"metrics.json").read_text())
        self.assertEqual(m["status"],"completed")
        x,train,test,texts=read_features()
        ix=len(train)
        for name in ("qwen","hash_bow"):
            from real_qwen_latent import hashed_bow
            features=x if name=="qwen" else hashed_bow(texts)
            file=out/(name+"_joint_student.pt")
            self.assertTrue(file.is_file())
            ckpt=torch.load(file,map_location="cpu",weights_only=False)
            self.assertIn("optimizer",ckpt)
            student=SemanticsFirstStudent()
            student.load_state_dict(ckpt["student"],strict=True)
            result,_=evaluate_joint(student,features[ix:ix+len(test)],test)
            self.assertEqual(result["paired_exact"],m["joint"][name]["heldout"]["paired_exact"])

if __name__=="__main__":unittest.main(verbosity=2)
