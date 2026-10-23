"""Diagnostics use only newly frozen development records, never the test set."""
import collections,json,math,re,time
import torch
from common import ROOT,setup,rows,load,encode,atomic_json,sha,score_labels
from tokenizer import Tokenizer,PAT
from model import Model,Config
def main():
    setup();tok=Tokenizer.load(ROOT/'data/tokenizer.json');dev=rows('dev_alpaca.jsonl');mcq=rows('dev_mcq.jsonl')
    texts=[r['prompt']+r['completion'] for r in dev+mcq];pieces=[m.group().encode() for text in texts for m in PAT.finditer(text)]
    encoded=[tok.encode(t) for t in texts]
    assert all(tok.decode(ids)==t for ids,t in zip(encoded,texts))
    stats={'vocab_size':tok.vocab_size,'tokenizer_sha256':sha(ROOT/'data/tokenizer.json'),'exact_roundtrip_all_dev':True,'total_tokens':sum(map(len,encoded)),'total_utf8_bytes':sum(len(t.encode()) for t in texts),'byte_fallback_token_fraction':sum(3<=i<259 for ids in encoded for i in ids)/sum(map(len,encoded)),'lexical_piece_exact_vocab_coverage':sum(p in tok.lookup for p in pieces)/len(pieces),'label_tokens':{chr(65+i):tok.encode(' '+chr(65+i),eos=True) for i in range(4)},'random_uniform_token_nll':math.log(tok.vocab_size)}
    models=[('L100EC_original',ROOT.parent/'ext_20261018/checkpoints/parent.bf16.safetensors'),('EXT_step256',ROOT.parent/'ext_20261018/checkpoints/step_0256/model.safetensors')]
    results={}
    for arm,weights in models:
        m,_=load(weights);m.eval();pred=[];losses=[];label_mass=[]
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
            for r in dev[:32]:
                x,y,n=encode(tok,r);losses.append((float(m(x,targets=y)),n))
            for r in dev[:8]:
                ids=torch.tensor([tok.encode(r['prompt'],bos=True)]);seq=m.generate(ids,32)[0].tolist();body=seq[:seq.index(2)] if 2 in seq else seq
                pred.append({'id':r['id'],'prompt':r['prompt'],'source_target':r['completion'],'generation':tok.decode(body),'tokens':seq})
            for r in mcq[:16]:
                plain=r['prompt'][len('User: '):].removesuffix('\nAssistant:')+'\nAnswer:'
                for name,prompt in [('matched',r['prompt']),('plain_diagnostic',plain)]:
                    rr=dict(r,prompt=prompt);scores,eos=score_labels(m,tok,rr)
                    label_mass.append({'id':r['id'],'format':name,'label_log_probs':scores,'gold':r['gold_index']})
        norms={name:{'norm':float(p.detach().float().norm()),'absmax':float(p.detach().float().abs().max())} for name,p in m.named_parameters() if name in ['embed.weight','norm.weight','blocks.0.norm1.weight','blocks.13.norm2.weight']}
        results[arm]={'weight_sha256':sha(weights),'strict_load_complete':True,'parameters':sum(p.numel() for p in m.parameters()),'finite_tensors':135,'dev32_nll':sum(l*n for l,n in losses)/sum(n for _,n in losses),'dev_generations':pred,'prompt_format_dev_diagnostic':label_mass,'selected_parameter_norms':norms}
        print(arm,'NLL',results[arm]['dev32_nll'],'example',repr(pred[0]['generation']),flush=True);del m
    # A no-training engineering reference quantifies the scale of a random model's
    # loss; not a competing trained model or a scientific result.
    torch.manual_seed(20261022);cfg=Config(**json.loads((ROOT/'checkpoints/config.json').read_text()));cfg.checkpoint_blocks=False;m=Model(cfg);m.eval();losses=[]
    with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
        for r in dev[:8]:
            x,y,n=encode(tok,r);losses.append((float(m(x,targets=y)),n))
    result={'tokenizer':stats,'models':results,'untrained_engineering_reference_dev8_nll':sum(l*n for l,n in losses)/sum(n for _,n in losses),'test_data_read_by_this_script':False,'optimizer_steps':0,'historical_evidence':'L100EC summary documents 1536 curriculum updates, 686613 input tokens and 203581 supervised tokens, 5676 unique synthetic curriculum IDs. Earlier full pretraining corpus/history unavailable; broad language pretraining is not established.'}
    atomic_json(ROOT/'metrics/diagnosis.json',result);print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
