from pathlib import Path
import copy,json,random,sys,tempfile
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'base_src')]
from data import solve,semantic_group,render,FAMILIES
from run import batch,generate_texts,load_rows
from model import Config,FlyGraphLM
from tokenizer import Tokenizer

def test_semantic_groups_disjoint():
    splits=['train','validation','test_iid','test_surface','test_extrapolation','test_counterfactual']
    groups=[{r['group'] for r in load_rows(s)} for s in splits]
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])

def test_all_gold_independently_replays():
    for split in ['train','validation','test_iid','test_surface','test_extrapolation','test_counterfactual']:
        for z in load_rows(split):
            c=z['case'];f=c['family']
            if f=='memory_update':
                want=next(v for k,v in reversed(c['writes']) if k==c['query'])
            elif f=='arithmetic':
                a,b=c['values'];op=c['op'];want=a+b if op=='+' else a-b if op=='-' else a*b
            elif f=='conditional':
                x,k,a,b=c['values'];want=x+(a if x<k else -b)
            elif f=='code_trace':
                want=c['start']
                for op,a in c['program']:want=want+a if op=='+' else want-a if op=='-' else want*a
            elif f=='list_reasoning':
                vs=sorted(c['values']);want=sum(vs) if c['kind']=='sum' else vs[0] if c['kind']=='min' else vs[-1]
            assert str(want)==z['answer']

def test_language_and_query_variants_cannot_cross_split():
    c={'family':'arithmetic','values':[11,22],'op':'+'}
    d=copy.deepcopy(c);d['op']='-'
    assert semantic_group(c)==semantic_group(d)
    c={'family':'memory_update','writes':[['a',3],['b',7]],'query':'a'}
    d=copy.deepcopy(c);d['query']='b'
    assert semantic_group(c)==semantic_group(d)

def test_loss_mask_alignment_and_inputs_identical():
    rr=load_rows('train')[:9]
    x,a=batch(rr,'answer_only');xx,b=batch(rr,'full_sequence');assert torch.equal(x,xx)
    for i,r in enumerate(rr):
        begin=r['prompt_tokens']-1;end=len(r['tokens'])-1
        assert (a[i,:begin]==-100).all()
        assert torch.equal(a[i,begin:end],b[i,begin:end])
        assert a[i,begin].item()==r['tokens'][r['prompt_tokens']]
        assert (a[i,end:]==-100).all() and (b[i,end:]==-100).all()

def test_counterfactual_pairs_different_answers():
    groups={}
    for z in load_rows('test_counterfactual'):groups.setdefault(z['pair_id'],[]).append(z)
    assert len(groups)==40
    assert all(len(g)==2 and g[0]['answer']!=g[1]['answer'] and g[0]['prompt']!=g[1]['prompt'] for g in groups.values())

def test_gold_fits_unrestricted_generation_budget():
    for s in ['test_iid','test_surface','test_extrapolation','test_counterfactual']:
        assert all(len(z['tokens'])-z['prompt_tokens']<=8 for z in load_rows(s))

def test_last_write_not_perfect_solution():
    zz=[z for z in load_rows('test_iid') if z['family']=='memory_update']
    rate=sum(str(z['case']['writes'][-1][1])==z['answer'] for z in zz)/len(zz)
    assert rate<.5

def test_tokenizer_fixed_roundtrip():
    t=Tokenizer.load(ROOT/'data/tokenizer.json');assert t.vocab_size==8192
    for s in ['ё 00123 = x\n\tprint(x)', '中文🙂 A != a']:
        assert t.decode(t.encode(s,bos=True,eos=True))==s

def tiny():
    torch.manual_seed(999)
    return FlyGraphLM(Config(vocab_size=300,d_model=32,n_layers=2,n_heads=4,n_kv_heads=2,d_ff=64,graph_width=8,checkpoint_blocks=False)).eval()

def test_causal_mask():
    m=tiny();x=torch.randint(3,300,(1,8));xx=x.clone();xx[:,4:]=torch.randint(3,300,(1,4))
    with torch.no_grad():assert torch.allclose(m(x)[:,:4],m(xx)[:,:4],atol=1e-6)

def test_cached_generation():
    m=tiny();x=torch.randint(3,300,(1,8))
    with torch.no_grad():
        a=m(x);_,cache=m(x[:,:5],use_cache=True);b,_=m(x[:,5:],past=cache,use_cache=True)
    assert torch.allclose(a[:,5:],b,atol=2e-5)

def test_prompt_only_interface():
    class Spy:
        def generate(self,ids,**kw):
            assert torch.equal(ids,self.expected);assert kw['max_new_tokens']==8
            return torch.cat([ids,torch.tensor([[tok.EOS]]*len(ids))],1)
    tok=Tokenizer.load(ROOT/'data/tokenizer.json');m=Spy();prompts=['Answer:','Answer:']
    m.expected=torch.tensor([tok.encode(p,bos=True) for p in prompts])
    assert generate_texts(m,tok,prompts)==[('',[],True),('',[],True)]

def test_batch_right_padding_does_not_change_prefix():
    m=tiny();x=torch.randint(3,300,(1,6));xx=torch.cat([x,torch.zeros((1,3),dtype=torch.long)],1)
    with torch.no_grad():assert torch.allclose(m(x),m(xx)[:,:6],atol=2e-5)
