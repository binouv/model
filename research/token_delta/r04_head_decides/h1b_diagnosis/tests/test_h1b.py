"""Run with python -m unittest discover -s h1b_diagnosis/tests -v ."""
import os
import sys
import unittest
from collections import Counter
from pathlib import Path

R04=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R04/'h1b_diagnosis'/'src'))
sys.path.insert(0,str(R04/'h1'/'src'))
from h1b_dataset import new_cases,one_step_question,sha_json,save_dataset,load_dataset
from h1_dataset import build_dataset,solve,verify_python
from h1b_run import parse_final,final_answer_segment,ARMS,score_rows
from h1b_audit import audit

class H1BDatasetTests(unittest.TestCase):
    def test_unique_and_outside_h1(self):
        h1_keys={x['group'] for x in build_dataset()}
        cases=new_cases()
        self.assertEqual(len(cases),128)
        self.assertEqual(len({c['group'] for c in cases}),128)
        self.assertFalse(h1_keys & {c['group'] for c in cases})
    def test_balanced_and_deterministic(self):
        c=new_cases()
        self.assertEqual(sha_json(c),sha_json(new_cases()))
        self.assertEqual(Counter((x['domain'],x['language']) for x in c),{
            ('arithmetic','en'):32,('arithmetic','ru'):32,
            ('code','en'):32,('code','ru'):32})
    def test_oracles_and_one_step_prompt(self):
        for r in new_cases():
            p,q=r['digits']
            self.assertEqual((p,q),solve(r['operator'],*r['operands']))
            self.assertEqual((p,q),verify_python(r['operator'],*r['operands']))
            question=one_step_question(r)
            self.assertNotIn('q = ',question)
            self.assertTrue(question.strip())
    def test_manifest_integrity(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            m=save_dataset(base)
            c,m2=load_dataset(base)
            self.assertEqual(len(c),128)
            self.assertEqual(m['new_dataset_sha256'],m2['new_dataset_sha256'])
            with (base/'h1b_cases.jsonl').open('a',encoding='utf-8') as f:f.write('X')
            with self.assertRaises(RuntimeError):load_dataset(base)

class ParserTests(unittest.TestCase):
    def test_pair_valid(self):
        for s in ('1 2', 'Answer: 1 2', 'ответ: 1 2.', '<|im_end|> 1 2 <|im_end|>',
                  'p=1, q=2','Результат:\n1 2'):
            self.assertEqual(parse_final(s,False)[0],[1,2],s)
    def test_avoid_numbers_inside_unclosed_thinking(self):
        out,status=parse_final('<think>p=1, q=2 but need compute',True)
        self.assertIsNone(out)
        self.assertEqual(status,'no_thinking_close_marker')
    def test_closed_thinking_only_final_segment(self):
        raw='<think>Maybe 9 9? More work. 5 5</think>\nФинал: 1 2<|im_end|>'
        self.assertEqual(parse_final(raw,True)[0],[1,2])
        self.assertEqual(parse_final('<think>9 9</think> No answer',True)[0],None)
    def test_one_digit_end(self):
        self.assertEqual(parse_final('Answer: 8.',False,True)[0],[8])
        self.assertIsNone(parse_final('Answer: 18.',False,True)[0])
    def test_do_not_parse_interior_digits(self):
        self.assertIsNone(parse_final('first 1 2, then 3 4, no answer',False)[0])
        self.assertIsNone(parse_final('<think>1 2</think>',True)[0])
    def test_arms_are_fixed(self):
        self.assertEqual(tuple(ARMS),('no_think_12','no_think_96','thinking_384','one_step_32'))
        self.assertEqual([v['max_tokens'] for v in ARMS.values()],[12,96,384,32])

class ScoringTests(unittest.TestCase):
    def test_full_grid_required(self):
        with self.assertRaises(ValueError):score_rows([],16)
    def test_baseline_audit_requires_300(self):
        with self.assertRaises(ValueError):audit([],build_dataset())

    def test_original_audit_is_read_only_and_exact(self):
        dataset=build_dataset()
        records=[{'id':r['id'],'oracle':r['digits'],'raw':r['answer'],'exact':True}
                  for r in dataset if r['split']=='test']
        outcome=audit(records,dataset)
        self.assertEqual(outcome['overall']['strict_exact_count'],300)
        self.assertEqual(outcome['overall']['invalid_format'],0)
        self.assertTrue(outcome['original_h1_gates_not_recomputed'])

    def test_full_grid_scoring_from_saved_raw_fields(self):
        rows=[]
        for i in range(16):
            for arm in ARMS:
                rows.append({'arm':arm,'id':f'h1b-{i:03d}', 'domain':'arithmetic',
                  'language':'en','operator':'add','semantic_exact':arm=='thinking_384',
                  'strict_exact':False, 'parsed':[1,2] if arm=='thinking_384' else None,
                  'truncated':False,'generated_tokens':4,'cuda_seconds':0.01,
                  'digit0_correct':int(arm=='thinking_384'),
                  'digit1_correct':int(arm=='thinking_384')})
        score=score_rows(rows,16)
        self.assertTrue(score['diagnostic_gates_predeclared']['serial_thinking_help'])
        self.assertEqual(score['arms']['thinking_384']['semantic_exact'],1.0)

if __name__=='__main__':unittest.main(verbosity=2)