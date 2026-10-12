"""CPU-only structural correctness for H2; NO pretrained quality claim."""
from __future__ import annotations
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
# The actual repo places latent_bridge.py at R04/src, not in H2.
sys.path.insert(1,str(ROOT.parent/'src'))
from h2_dataset import build,save,load,oracle,source_code,excluded_keys,canonical_sha
from h2_train import (lexical_features,oracle_feature,parse_digits,
    make_teacher_forcing_batch,shifted_ce,write_checkpoint,restore_checkpoint,
    score_predictions,read_cache,generate_writer)
from latent_bridge import LatentBridge

class H2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.rows=build()
    def test_locked_size_splits(self):
        rows=self.rows
        self.assertEqual(len(rows),1700)
        self.assertEqual([sum(x['split']==s for x in rows) for s in ('train','val','test')],[1200,200,300])
        self.assertEqual(len(set(x['id'] for x in rows)),1700)
        self.assertEqual(len(set(x['group'] for x in rows)),1700)
    def test_exclusion_of_old_h1_h1b_h1c(self):
        old=excluded_keys()
        self.assertEqual(len(old),3528)
        for row in self.rows:
            self.assertNotIn((row['operator'],*row['operands']),old)
    def test_oracle_matches_python(self):
        for row in self.rows:
            self.assertEqual(tuple(row['digits']),oracle(row['operator'],*row['operands']))
            code=source_code(row['operator'],*row['operands'])
            # Fixed generated program; no user content is evaluated.
            stream=io.StringIO()
            from contextlib import redirect_stdout
            with redirect_stdout(stream):
                exec(compile(code,'<fixed-generated>', 'exec'),{'__builtins__':{'print':print}}, {})
            self.assertEqual(stream.getvalue().strip(),row['answer'])
    def test_manifest_roundtrip_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            fp=Path(td)
            manifest=save(fp)
            saved,_=load(fp)
            self.assertEqual(canonical_sha(saved),manifest['canonical_sha256'])
            data=fp/'h2_dataset.jsonl'
            data.write_bytes(data.read_bytes()+b'\n')
            with self.assertRaises(RuntimeError):load(fp)
    def test_strict_scoring_and_controls(self):
        self.assertEqual(parse_digits('1 3'),[1,3])
        self.assertIsNone(parse_digits('Answer: 1 3'))
        self.assertIsNone(parse_digits('1 3 5'))
        rows=self.rows[:2]
        teacher=[{'teacher_plain':'2 3','teacher_correct':True},
                 {'teacher_plain':'wrong','teacher_correct':False}]
        predictions=[{'writer_plain':rows[0]['answer'],'id':rows[0]['id']},
                     {'writer_plain':'wrong','id':rows[1]['id']}]
        teacher[0]['teacher_correct']=False
        metrics=score_predictions(rows,teacher,predictions)
        self.assertEqual(metrics['n'],2)
        self.assertIsNone(metrics['writer_accuracy_on_teacher_correct'])
    def test_input_controls_do_not_see_gold_labels(self):
        rows=self.rows[:16]
        f=lexical_features(rows)
        o=oracle_feature(rows)
        self.assertEqual(f.shape,(16,2560))
        self.assertEqual(o.shape,(16,2560))
        changed=[dict(x,answer='wrong',digits=[9,9]) for x in rows]
        self.assertTrue(torch.equal(f,lexical_features(changed)))
        self.assertFalse(torch.equal(o,oracle_feature(changed)))
    def test_causal_alignment_and_masking(self):
        b,k,w=2,2,16
        embed=nn.Embedding(20,w)
        z=torch.randn(b,k,w,requires_grad=True)
        header=torch.tensor([[2],[2]])
        emb,mask,gold=make_teacher_forcing_batch(z,header,[[3,4],[5]],embed)
        self.assertEqual(emb.shape,(b,k+1+2,w))
        self.assertEqual(mask.shape,(b,k+1+2))
        self.assertEqual(gold.tolist(),[[3,4],[5,-100]])
        # Logit t=k predicts target[0]. Logit t=k+1 predicts target[1].
        logits=torch.full((b,k+1+2,20),-9.0)
        logits[0,k,3]=9.0
        logits[0,k+1,4]=9.0
        logits[1,k,5]=9.0
        self.assertLess(float(shifted_ce(logits,gold,k,1)),1e-5)
        logits[:,k+2,:]=-1e5
        self.assertLess(float(shifted_ce(logits,gold,k,1)),1e-5)
    def test_softprefix_receives_gradient_but_frozen_writer_does_not(self):
        bridge=LatentBridge(32,16,2,24)
        embed=nn.Embedding(20,16)
        head=nn.Linear(16,20)
        embed.requires_grad_(False)
        head.requires_grad_(False)
        x=torch.randn(2,32)
        z=bridge(x)
        inputs,mask,gold=make_teacher_forcing_batch(z,torch.tensor([[1],[1]]),[[3,4],[5]],embed)
        # A causal toy writer must couple current output to earlier latent slots.
        logits=head(torch.cumsum(inputs,dim=1))
        loss=shifted_ce(logits,gold,2,1)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.norm()>0 for p in bridge.parameters()))
        self.assertTrue(all(p.grad is None for p in embed.parameters()))
        self.assertTrue(all(p.grad is None for p in head.parameters()))
    def test_manual_writer_soft_prefix_only(self):
        from types import SimpleNamespace
        class FrozenWriter(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb=nn.Embedding(32,8)
                self.emb.requires_grad_(False)
            def get_input_embeddings(self):return self.emb
            def forward(self,inputs_embeds,use_cache=False,return_dict=True):
                # Only embedded prefix+constant header reaches the model.
                out=torch.zeros(inputs_embeds.size(0),inputs_embeds.size(1),32)
                out[:,-1,7]=100.0
                return SimpleNamespace(logits=out)
        class ToyTok:
            def convert_tokens_to_ids(self,token):
                return {'<|im_end|>':7,'<|endoftext|>':8}[token]
            def decode(self,token_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False):
                return '' if skip_special_tokens else '<|im_end|>'
        writer=FrozenWriter().eval()
        z=torch.randn(1,4,8)
        raw,plain,ids=generate_writer(writer,ToyTok(),z,torch.tensor([[1]]),mode='manual')
        self.assertEqual(ids,[7])
        self.assertEqual(raw,'<|im_end|>')
        self.assertEqual(plain,'')
    def test_checkpoint_small_unsharded_and_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            old={'hello':'world','weights':torch.arange(8)}
            manifest=write_checkpoint(Path(td),'small.pt',old)
            self.assertEqual(len(manifest['parts']),1)
            self.assertEqual(restore_checkpoint(Path(td),manifest)['hello'],'world')

    def test_checkpoint_exact_bytes_and_lossless_split(self):
        import h2_train
        old=h2_train.SPLIT_BYTES
        with tempfile.TemporaryDirectory() as td:
            try:
                h2_train.SPLIT_BYTES=512
                source={'weights':torch.arange(400,dtype=torch.float32),
                        'optimizer':{'test_step':7},'rng':torch.get_rng_state()}
                receipt=write_checkpoint(Path(td),'toy.pt',source)
                self.assertGreater(len(receipt['parts']),1)
                loaded=restore_checkpoint(Path(td),receipt)
                self.assertTrue(torch.equal(loaded['weights'],source['weights']))
                self.assertEqual(loaded['optimizer']['test_step'],7)
                first=Path(td)/receipt['parts'][0]['name']
                first.write_bytes(first.read_bytes()+b'corrupt')
                with self.assertRaises(RuntimeError):restore_checkpoint(Path(td),receipt)
            finally:h2_train.SPLIT_BYTES=old
    def test_pilot_cache_hash_and_strict_order(self):
        from h2_dataset import sha256_file
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            data=td/'data';save(data)
            rows,_=load(data)
            train=[x for x in rows if x['split']=='train'][:4]
            from safetensors.torch import save_file
            folder=td/'cache';folder.mkdir()
            spec={'head_last':torch.randn(4,2560,dtype=torch.bfloat16)}
            save_file(spec,str(folder/'h2_head_features.safetensors'),metadata={
                'dataset_sha256':canonical_sha(rows),'head_revision':
                'c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97',
                'pilot':'true','count':'4'})
            records=[]
            for x in train:
                records.append({'id':x['id'],'oracle':x['digits'],'split':'train',
                                'teacher_plain':'1 2','teacher_correct':False})
            file=folder/'teacher_outputs.jsonl'
            file.write_text(''.join(json.dumps(r)+'\n' for r in records))
            receipt={'pilot':True,'head_revision':'c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97',
                    'dataset_sha256':canonical_sha(rows),'features_sha256':sha256_file(folder/'h2_head_features.safetensors'),
                    'teacher_outputs_sha256':sha256_file(file)}
            (folder/'feature_manifest.json').write_text(json.dumps(receipt))
            selected,out,z,_=read_cache(folder,rows,canonical_sha(rows),True)
            self.assertEqual(len(out),4)
            self.assertEqual(z.shape,(4,2560))
            file.write_text('corruption')
            with self.assertRaises(RuntimeError):read_cache(folder,rows,canonical_sha(rows),True)

if __name__=='__main__':unittest.main(verbosity=2)
