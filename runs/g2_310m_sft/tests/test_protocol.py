import copy,hashlib,json,operator,random,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'base_src')]
from data import solve,FAMILIES,semantic_group
from run import batch,load_rows
from model import Config,FlyGraphLM,parameter_report
from tokenizer import Tokenizer
torch.set_num_threads(1)

def allrows():
    return {p.stem:[json.loads(x) for x in p.read_text().splitlines()] for p in (ROOT/'data').glob('*.jsonl')}

def test_gold_independent_executor():
    ops={'+':operator.add,'-':operator.sub,'*':operator.mul}
    for rows in allrows().values():
        for z in rows:
            c=z['case'];f=c['family']
            if f=='arithmetic':a=ops[c['op']](*c['values'])
            elif f=='conditional':
                x,k,u,d=c['values'];a=(x+u,x-d)[int(x>=k)]
            elif f=='code_trace':
                a=c['start']
                for op,n in c['program']:a=ops[op](a,n)
            elif f=='list_reasoning':a={'sum':sum,'min':min,'max':max}[c['kind']](c['values'])
            else:a=next(v for k,v in reversed(c['writes']) if k==c['query'])
            assert str(a)==z['answer']

def test_semantic_groups_disjoint():
    rows=allrows();groups={s:{z['group'] for z in rr} for s,rr in rows.items()}
    for s in groups:
        for t in groups:
            if s!=t:assert not groups[s]&groups[t]

def test_language_not_task_identifier():
    rows=load_rows('train')
    for f in FAMILIES:
        assert {z['lang'] for z in rows if z['family']==f}=={'ru','en'}

def test_counterfactual_pairs():
    pairs={}
    for z in load_rows('test_counterfactual'):pairs.setdefault(z['pair_id'],[]).append(z)
    assert len(pairs)==40
    for pair in pairs.values():
        assert len(pair)==2 and pair[0]['answer']!=pair[1]['answer']
        assert semantic_group(pair[0]['case'])==semantic_group(pair[1]['case'])

def test_complete_answer_masks():
    tok=Tokenizer.load(ROOT/'data/tokenizer.json')
    rows=load_rows('train')[:12]
    x,y,mask=batch(rows,'answer_only');xx,yy,mm=batch(rows,'full_sequence')
    assert torch.equal(x,xx) and torch.equal(mask,mm)
    for i,z in enumerate(rows):
        ct=tok.encode(z['completion'],eos=True)
        assert y[i,z['prompt_tokens']-1:z['prompt_tokens']-1+len(ct)].tolist()==ct
        assert y[i,:z['prompt_tokens']-1].eq(-100).all()
        assert y[i].ne(-100).sum()==len(ct)

def test_no_truncation():
    for split in ['train','validation']:
        for z in load_rows(split):assert len(z['tokens'])<=96 and z['tokens'][-1]==2

def test_gold_not_in_prompt_slice():
    tok=Tokenizer.load(ROOT/'data/tokenizer.json')
    for z in load_rows('test_iid'):
        assert z['tokens'][:z['prompt_tokens']]==tok.encode(z['prompt'],bos=True)
        zz=copy.deepcopy(z);zz['case']={};zz['answer']='WRONG';zz['completion']='WRONG'
        zz['tokens'][zz['prompt_tokens']:]=[3,4,5]
        assert zz['tokens'][:zz['prompt_tokens']]==z['tokens'][:z['prompt_tokens']]

def test_full_vocab_and_parameter_count():
    with torch.device('meta'):m=FlyGraphLM(Config())
    assert m.config.vocab_size==8192 and parameter_report(m)['trainable_parameters']==309993253

def test_right_padding_does_not_change_prefix():
    torch.manual_seed(91);m=FlyGraphLM(Config(vocab_size=300,d_model=64,n_layers=4,n_heads=4,n_kv_heads=2,d_ff=128,graph_width=16,checkpoint_blocks=False)).eval()
    x=torch.randint(3,300,(1,8));p=torch.cat([x,torch.zeros((1,4),dtype=torch.long)],1)
    with torch.no_grad():assert torch.allclose(m(x),m(p)[:,:8],atol=2e-5)

def test_checkpoint_meta_matches_prereg():
    z=json.loads((ROOT/'configs/preregistered.json').read_text())
    assert z['steps_per_arm']==256 and z['architecture_changed'] is False and z['exact_G1_resume'] is False

def test_no_constant_memory_shortcut():
    z=json.loads((ROOT/'data/manifest.json').read_text())
    assert z['memory_last_global_write_shortcut']<.4
