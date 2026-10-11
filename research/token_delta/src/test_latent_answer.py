"""Formal unit checks for the synthetic research interface and published checkpoints."""
import json
import os
import unittest
from pathlib import Path
import torch
from latent_answer import (AGENTS,VERBS,OBJECTS,VOCAB,TOKEN,EOS,MAX_LEN,
                           LatentSystem,Planner,Renderer,split,grammar,evaluate,make_batch,readable)

class TestRepresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
    def test_dataset_no_combination_leakage(self):
        train,test=split(20261012)
        self.assertEqual(len(train),410)
        self.assertEqual(len(test),102)
        self.assertEqual(len(set(train)),410)
        self.assertFalse(set(train).intersection(set(test)))
        for dim in range(3):
            self.assertEqual(set(x[dim] for x in train),set(range(8)))
            self.assertEqual(set(x[dim] for x in test),set(range(8)))
    def test_semantic_plan_independent_of_style(self):
        torch.manual_seed(13)
        model=Planner()
        x=torch.tensor([(0,3,5),(4,7,2)],dtype=torch.long)
        lat=model(x)
        self.assertEqual(lat.shape,(2,32))
        self.assertTrue(torch.equal(lat,model(x)))
    def test_decoder_uses_both_styles(self):
        facts=(2,5,7)
        short=grammar(facts,0)
        detail=grammar(facts,1)
        self.assertNotEqual(short,detail)
        self.assertEqual(short[-1],EOS)
        self.assertEqual(detail[-1],EOS)
        for w in (AGENTS[facts[0]],VERBS[facts[1]],OBJECTS[facts[2]]):
            self.assertIn(w,readable(short))
            self.assertIn(w,readable(detail))
    def test_batch_shapes(self):
        x,style,inp,target=make_batch([((0,1,2),0),((1,2,3),1)])
        self.assertEqual(tuple(inp.shape),(2,MAX_LEN))
        self.assertEqual(tuple(target.shape),(2,MAX_LEN))
        self.assertEqual(style.tolist(),[0,1])
        model=LatentSystem()
        with torch.no_grad():
            logits=model(x,inp,style)
        self.assertEqual(tuple(logits.shape),(2,MAX_LEN,len(VOCAB)))
    def test_checkpoint_reload_and_accuracy(self):
        checkpoint = Path(os.environ.get('TOKEN_DELTA_CKPT','results/latent_answer_training_checkpoint.pt'))
        if not checkpoint.is_file():
            self.skipTest(f'optional training checkpoint unavailable: {checkpoint}')
        raw=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=LatentSystem()
        model.planner.load_state_dict(raw['planner'],strict=True)
        model.renderer.load_state_dict(raw['renderer'],strict=True)
        expected=raw['test_facts']
        metrics,_=evaluate(model,expected)
        self.assertEqual(metrics['paired_exact'],1.0)
        reduced,_=evaluate(model,expected,'zero')
        self.assertLess(reduced['paired_exact'],metrics['paired_exact'])

if __name__=='__main__':unittest.main(verbosity=2)
