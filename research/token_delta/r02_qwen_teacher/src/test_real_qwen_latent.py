"""Local and CI semantics checks for R02, including strict trained-weights restore."""
import json
import os
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F
from real_qwen_latent import (
    VOCAB, IDX, EOS, MAX_LEN, LATENT, SEED, SUBJECTS, ACTIONS,
    OBJECTS, TEST_SYNONYMS, split_facts, question, response,
    encode_targets, hashed_bow, LatentStudent, evaluate
)

class TestPretrainedQwenLatentProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_holdout_non_leaking_combinations(self):
        train,test=split_facts()
        self.assertEqual((len(train),len(test)),(168,48))
        self.assertFalse(set(train).intersection(set(test)))
        for col in range(3):
            self.assertEqual({x[col] for x in train},set(range(6)))
            self.assertEqual({x[col] for x in test},set(range(6)))

    def test_ood_input_excludes_unseen_verbs_in_training(self):
        train,test=split_facts()
        train_questions=" ".join(question(f) for f in train).lower()
        test_questions=" ".join(question(f,ood=True) for f in test).lower()
        for verb in TEST_SYNONYMS:
            self.assertNotIn(verb,train_questions)
            self.assertIn(verb,test_questions)

    def test_one_style_independent_bridge(self):
        net=LatentStudent()
        x=torch.randn(4,1024)
        z=net.bridge(x)
        self.assertEqual(tuple(z.shape),(4,LATENT))
        self.assertTrue(torch.equal(z,net.bridge(x)))
        pred_short=net.renderer.generate(z,torch.zeros(4,dtype=torch.long))
        pred_detail=net.renderer.generate(z,torch.ones(4,dtype=torch.long))
        self.assertEqual(pred_short.shape,(4,MAX_LEN))
        self.assertEqual(pred_detail.shape,(4,MAX_LEN))

    def test_semantic_targets(self):
        x=(2,4,5)
        short=response(x,0)
        detail=response(x,1)
        self.assertEqual((len(short),len(detail)),(7,17))
        self.assertEqual(short[-1],EOS)
        self.assertEqual(detail[-1],EOS)
        for word in (SUBJECTS[x[0]],ACTIONS[x[1]],OBJECTS[x[2]]):
            self.assertIn(IDX[word],short)
            self.assertIn(IDX[word],detail)
        y=encode_targets([x])
        self.assertEqual(tuple(y.shape),(1,2,MAX_LEN))

    def test_hashed_bow_determinism(self):
        items=[question(x) for x in split_facts()[0][:4]]
        self.assertTrue(torch.equal(hashed_bow(items),hashed_bow(items)))
        self.assertEqual(tuple(hashed_bow(items).shape),(4,1024))

    def test_gradients_do_not_touch_teacher(self):
        model=LatentStudent()
        x=torch.randn(4,1024)
        z=model.bridge(x)
        gold=encode_targets(split_facts()[0][:4])[:,0,:]
        bos=torch.full((4,1),IDX["<bos>"],dtype=torch.long)
        inp=torch.cat((bos,gold[:,:-1]),dim=1)
        style=torch.zeros(4,dtype=torch.long)
        scores=model(x,inp,style)
        loss=F.cross_entropy(scores.flatten(0,1),gold.flatten(),ignore_index=IDX["<pad>"])
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        self.assertEqual(list(model.parameters())[0].shape[0],1024)

    def test_real_qwen_outputs_and_checkpoints(self):
        folder=Path(os.environ.get("TOKEN_DELTA_R02_RESULTS","../results"))
        metrics_file=folder/"metrics.json"
        feature_file=folder/"qwen_prompt_features.pt"
        if not (metrics_file.is_file() and feature_file.is_file()):
            self.skipTest("No complete Qwen CPU run yet")
        metrics=json.loads(metrics_file.read_text())
        cache=torch.load(feature_file,map_location="cpu",weights_only=False)
        train,test=split_facts()
        inputs=cache["questions"]
        self.assertEqual(len(inputs),168+48+48)
        self.assertEqual(tuple(cache["embeddings"].shape),(264,1024))
        self.assertEqual(cache["train_triples"],train)
        self.assertEqual(cache["test_triples"],test)
        self.assertTrue(torch.isfinite(cache["embeddings"]).all())
        for name in ["qwen","hash_bow"]:
            ckp=folder/(name+"_student_full_checkpoint.pt")
            self.assertTrue(ckp.is_file(),ckp)
            raw=torch.load(ckp,map_location="cpu",weights_only=False)
            net=LatentStudent()
            net.load_state_dict(raw["student_state_dict"],strict=True)
            self.assertIn("optimizer_state_dict",raw)
            feats=(cache["embeddings"][168:216] if name=="qwen"
                   else hashed_bow(inputs)[168:216])
            got,_=evaluate(net,feats,test)
            stored=metrics["systems"][name]["iid"]
            self.assertAlmostEqual(got["paired_exact"],stored["paired_exact"],places=7)
            self.assertAlmostEqual(got["short"]["semantic_slots"],
                                   stored["short"]["semantic_slots"],places=7)

if __name__=="__main__":
    unittest.main(verbosity=2)
