"""CPU-only engineering tests; not the scientific H1 experiment."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from h1_dataset import (
    build_dataset,dataset_sha256,materialize,load_dataset,solve,python_code,
    verify_python,question,strict_parse_answer,
)
from h1_probes import TwoDigitProbe, lexical_features, structured_features, classify


class TestH1Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all=build_dataset()
    def test_1700_and_stratified_splits(self):
        self.assertEqual(len(self.all),1700)
        self.assertEqual(Counter(x['split'] for x in self.all),{'train':1200,'val':200,'test':300})
        buckets=Counter((x['domain'],x['language'],x['split']) for x in self.all)
        self.assertEqual(len(buckets),12)
        for (domain,lang,split),n in buckets.items():
            self.assertEqual(n,{'train':300,'val':50,'test':75}[split])
    def test_no_duplicate_problem_or_hidden_answer_as_assistant(self):
        self.assertEqual(len({x['group'] for x in self.all}),len(self.all))
        for x in self.all:
            self.assertNotIn('assistant',x['question'].lower())
            self.assertEqual(x['digits'],list(solve(x['operator'],*x['operands'])))
            self.assertTrue('?' in x['question'] or 'ЦИФРЫ' in x['question'] or 'DIGITS' in x['question'])
    def test_code_oracle_matches_actual_python_program(self):
        for x in self.all:
            if x['domain']=='code':
                self.assertEqual(verify_python(x['operator'],*x['operands']),tuple(x['digits']))
    def test_dataset_persisted_hash_and_byte_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)
            original=materialize(dest)
            data,manifest=load_dataset(dest)
            self.assertEqual(original['dataset_sha256_canonical'],dataset_sha256(data))
            with (dest/'h1_dataset.jsonl').open('a',encoding='utf-8') as out:out.write(' ')  # modifies bytes
            with self.assertRaises(RuntimeError):load_dataset(dest)
    def test_strict_answer_parser(self):
        self.assertEqual(strict_parse_answer(' 3 7 '),[3,7])
        self.assertEqual(strict_parse_answer('2\n9'),[2,9])
        self.assertIsNone(strict_parse_answer('Answer: 2 9'))
        self.assertIsNone(strict_parse_answer('2 9 because...'))
        self.assertIsNone(strict_parse_answer('20 9'))
        self.assertIsNone(strict_parse_answer('2'))


class TestH1Probes(unittest.TestCase):
    def test_feature_shapes_and_determinism(self):
        rows=build_dataset()[:7]
        a=lexical_features(rows)
        b=lexical_features(rows)
        self.assertEqual(tuple(a.shape),(7,2560))
        self.assertTrue(torch.equal(a,b))
        strong=structured_features(rows)
        self.assertEqual(tuple(strong.shape),(7,66))
        self.assertTrue(torch.allclose(strong.sum(-1),torch.full((7,),6.)))
    def test_probe_two_digits_and_backprop(self):
        for arch in ('linear','mlp'):
            mdl=TwoDigitProbe(66,arch)
            x=torch.rand(8,66)
            expected=torch.randint(0,10,(8,2))
            scores=mdl(x)
            self.assertEqual(tuple(scores.shape),(8,2,10))
            loss=torch.nn.functional.cross_entropy(scores.reshape(-1,10),expected.reshape(-1))
            loss.backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in mdl.parameters()))
    def test_classification_reports_joint_exact(self):
        rows=build_dataset()[:9]
        input_=structured_features(rows)
        model=TwoDigitProbe(66,'linear')
        gold=torch.tensor([r['digits'] for r in rows])
        result,pred=classify(model,input_,gold,list(range(9)),rows)
        self.assertEqual(result['n'],9)
        self.assertEqual(len(pred),9)
        self.assertTrue(0<=result['exact']<=1)
        self.assertTrue(0<=result['digit1']<=1)


if __name__=='__main__': unittest.main(verbosity=2)