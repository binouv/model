"""Checkpoint replay and random negative-control verification."""
from pathlib import Path
import json
import os
import unittest
import numpy as np
import torch
from tiny_future import TinyGRU, FutureProbe, batch, CONTEXT, K, VOCAB

class TestFutureProbe(unittest.TestCase):
    def test_checkpoint_replay(self):
        base=Path(os.environ.get('TOKEN_DELTA_RESULTS','../results'))
        if not (base/'metrics.json').exists():
            self.skipTest('no training results found')
        expected=json.loads((base/'metrics.json').read_text())
        torch.set_num_threads(2)
        for task in ('motif','random'):
            path=base/f'{task}_training_checkpoint.pt'
            with self.subTest(task=task):
                self.assertTrue(path.is_file())
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                self.assertEqual(checkpoint['task'],task)
                self.assertIn('model_optimizer',checkpoint)
                self.assertIn('probe_optimizer',checkpoint)
                m=TinyGRU(hidden=checkpoint['config']['hidden']).eval()
                p=FutureProbe(hidden=checkpoint['config']['hidden']).eval()
                m.load_state_dict(checkpoint['model'],strict=True)
                p.load_state_dict(checkpoint['probe'],strict=True)
                b=batch(task,checkpoint['config']['eval_samples'],np.random.default_rng(24+9439))
                with torch.no_grad():
                    _,states,_=m(b[:,:CONTEXT])
                    top=p(states[:,-1]).argmax(-1)
                acc=(top==b[:,CONTEXT:CONTEXT+K]).float().mean(0).tolist()
                for x,y in zip(acc,expected[task]['probe_horizon_accuracy']):
                    self.assertAlmostEqual(x,y,places=7)
                mean_acc=float(np.mean(acc))
                if task=='motif': self.assertGreater(mean_acc,0.9)
                else: self.assertLess(abs(mean_acc-1/VOCAB),0.02)

if __name__=='__main__':unittest.main(verbosity=2)
