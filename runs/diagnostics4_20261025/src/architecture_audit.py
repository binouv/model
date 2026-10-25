"""Count current/future configurations on meta, and measure existing tokenizer."""
import collections,dataclasses,datetime,json,math
import torch
from common import ROOT,atomic_json,rows
from model import Model,Config
from tokenizer import Tokenizer

def main():
 cfg=Config(**json.loads((ROOT/'checkpoints/config.json').read_text()))
 with torch.device('meta'):m=Model(cfg)
 counts=collections.Counter()
 for n,p in m.named_parameters():
  key='embedding' if n=='embed.weight' else 'final_norm' if n=='norm.weight' else ('delta_mix' if m.blocks[int(n.split('.')[1])].kind=='d' else 'attention_mix') if '.mix.' in n else 'ffn' if '.uv.' in n or '.down.' in n else 'block_norm'
  counts[key]+=p.numel()
 variants=[]
 for layers,vocab,dim,heads,kv,hidden,name in [(14,8192,768,12,4,2048,'current'),(14,32768,768,12,4,2048,'same_backbone_vocab32k'),(28,8192,768,12,4,2048,'depth28_meta_only'),(42,8192,768,12,4,2048,'depth42_meta_only'),(30,32768,576,9,3,1536,'deep_thin_hybrid_meta_only')]:
  c=dataclasses.replace(cfg,layers=layers,vocab_size=vocab,dim=dim,heads=heads,kv_heads=kv,hidden=hidden)
  with torch.device('meta'):v=Model(c)
  variants.append(dict(name=name,config=dataclasses.asdict(c),parameters=sum(p.numel() for p in v.parameters()),trained=False if name!='current' else 'existing checkpoint; not trained by this script'))
 tok=Tokenizer.load(ROOT/'data/tokenizer.json');stats={}
 for task in ['openbookqa','piqa','arc_challenge']:
  rr=[r for r in rows('dev_mcq.jsonl') if r['task']==task];text=[r['question']+' '+ ' '.join(r['choices']) for r in rr];tt=[tok.encode(s) for s in text];flat=sum(tt,[])
  # This tokenizer reserves3 special IDs, then256 byte IDs; lexical IDs start259.
  stats[task]=dict(n=len(rr),tokens=len(flat),utf8_bytes=sum(len(s.encode()) for s in text),characters=sum(len(s) for s in text),byte_fallback_tokens=sum(3<=x<259 for x in flat),byte_fallback_fraction=sum(3<=x<259 for x in flat)/len(flat),all_exact_roundtrips=all(tok.decode(t)==s for t,s in zip(tt,text)))
 ext2=json.loads((ROOT.parent/'ext2_20261022/metrics/final_results.json').read_text());wall=ext2['training']['wall_seconds'] if 'training' in ext2 else None
 atomic_json(ROOT/'metrics/architecture_audit.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),config=dataclasses.asdict(cfg),parameters=sum(counts.values()),components=dict(counts),block_pattern='da'*7,parameter_tensors=len(list(m.parameters())),delta_state_bytes_per_sequence=7*12*32*64*4,attention_KV_bytes_at_context2048_BF16=7*2*4*2048*64*2,variants_meta_only=variants,tokenizer_new_dev=stats,nominal_zero_gate_half_life=math.log(2)/torch.nn.functional.softplus(torch.tensor(-4.)).item(),exact_label_token_ids={chr(65+i):tok.encode(' '+chr(65+i)) for i in range(4)},architecture_limits=['multi-token delta_rule materializes TxT matrices and a triangular solve','Model.forward context accounting assumes first attention cache at index1','Attention uses repeated KV tensors before SDPA','no QK-RMSNorm in attention; L2 q/k normalization in Delta is different','no MoE, no retrieval, no convolution in Delta'],inference_cache_scopes='State/KV bytes exclude other activations, repeated KV and logits; full attention KV still grows with context.'))
 print(json.dumps(dict(parameters=sum(counts.values()),components=dict(counts),tokenizer=stats,variants=[(v['name'],v['parameters']) for v in variants]),indent=2))
if __name__=='__main__':main()
