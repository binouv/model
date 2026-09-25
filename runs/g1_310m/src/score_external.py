"""Offline scoring contract for a future genuinely paired external-model run.
This script does not fetch or run Qwen and cannot fabricate its results.
Each prediction row must include id, model_id, checkpoint_revision, completion,
max_new_tokens, temperature, and tool_access. Keep raw model responses.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path

def norm(s):return re.sub(r'\s+',' ',s.strip()).rstrip('.')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gold',required=True);ap.add_argument('--predictions',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    gold={r['id']:r for r in map(json.loads,Path(args.gold).read_text().splitlines())}
    preds=list(map(json.loads,Path(args.predictions).read_text().splitlines()))
    ids=[z['id'] for z in preds]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate prediction IDs')
    if set(ids)!=set(gold):raise ValueError('missing or unexpected predictions; no partial aggregate')
    rows=[]
    for p in preds:
        for key in ['model_id','checkpoint_revision','completion','max_new_tokens','temperature','tool_access']:
            if key not in p:raise ValueError('missing inference metadata: '+key)
        g=gold[p['id']];rows.append({'id':p['id'],'strict_completion':norm(p['completion'])==norm(g['gold_completion'])})
    out={'status':'completed','n':len(rows),'strict_completion_accuracy':sum(x['strict_completion'] for x in rows)/len(rows),'rows':rows,'important':'Internal synthetic suite only; not an external broad reasoning benchmark. Final-answer accuracy must use the exact parser from evaluate_release.py.'}
    Path(args.out).write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
