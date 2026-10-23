"""Explain ranking/generation disagreements from saved answers; no inference."""
import json
from pathlib import Path
from tokenizer import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
def main():
    rows=[json.loads(s) for s in (ROOT/'metrics/selected_EXT2_test_predictions.jsonl').read_text().splitlines()]
    tok=Tokenizer.load(ROOT/'data/tokenizer.json');tokens={c:tok.encode(' '+c) for c in 'ABCD'}
    assert all(len(v)==1 for v in tokens.values())
    out=[]
    for p in rows:
        if p['parsed_generation']==p['predicted_label']:continue
        q=p['parsed_generation'];scores=p['label_log_probs']
        out.append({'id':p['id'],'task':p['task'],'label_prediction':p['predicted_label'],'generation_prediction':q,'scores':scores,'first_new_token':p['new_token_ids'][0],'top_score_tie':q is not None and scores[q]==max(scores)})
    result={'different_label_and_generation_predictions':len(out),'all_disagreements_are_exact_top_score_ties':all(p['top_score_tie'] for p in out),'label_ranking_tie_break':'first option in A/B/C/D order','full_vocab_greedy_tie_break':'lowest vocabulary token ID','label_token_ids':{k:v[0] for k,v in tokens.items()},'rows':out,'selection_or_evaluation_protocol_modified':False}
    (ROOT/'metrics/label_generation_tie_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':print(json.dumps(main(),indent=2))
