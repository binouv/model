"""Offline demonstration of pretrained Qwen semantic latent -> 2 response styles.

Loads *real* Qwen3-0.6B prompt features saved in verified R02, and the
trained R03 semantic-bottleneck student. Does not call Qwen again.
Can replay the same heldout prompt with either or both styles.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from probe_and_decode import (
    R02_ROOT, read_features, SemanticsFirstStudent,
    SUBJECTS, ACTIONS, OBJECTS, show_text, response, SEED
)

HERE=Path(__file__).resolve()
DEFAULT_CKPT=HERE.parents[1]/"results"/"qwen_joint_student.pt"

@torch.inference_mode()
def run_example(index=0, kind="ood", checkpoint=DEFAULT_CKPT):
    if kind not in ("iid","ood"):
        raise ValueError("kind must be iid or ood")
    if not 0<=index<48:
        raise ValueError("index must be between 0 and 47")
    features,train,test,questions=read_features()
    # Precomputed and independently sha-checked official Qwen3-0.6B states.
    offset=len(train)+(0 if kind=="iid" else len(test))
    f=features[offset+index:offset+index+1]
    data=torch.load(checkpoint,map_location="cpu",weights_only=False)
    net=SemanticsFirstStudent().eval()
    net.load_state_dict(data["student"],strict=True)
    logits,z=net.features(f)
    slots=logits.argmax(-1)[0].tolist()
    generated={}
    for i,label in enumerate(("short","detailed")):
        pred=net.renderer.generate(z,torch.tensor([i],dtype=torch.long))[0].tolist()
        generated[label]=show_text(pred)
    facts=test[index]
    return {
       "experiment":"R03 frozen real Qwen3-0.6B latent",
       "teacher_forward_calls_in_demo":0,
       "precomputed_frozen_teacher_question":questions[offset+index],
       "checkpoint_sha256":hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
       "latent_dimensions":int(z.shape[-1]),
       "latent_built_once":True,
       "predicted_facts":[SUBJECTS[slots[0]],ACTIONS[slots[1]],OBJECTS[slots[2]]],
       "gold_facts":[SUBJECTS[facts[0]],ACTIONS[facts[1]],OBJECTS[facts[2]]],
       "short":generated["short"],"detailed":generated["detailed"],
       "gold_short":show_text(response(facts,0)),
       "gold_detailed":show_text(response(facts,1)),
       "both_exact":generated["short"]==show_text(response(facts,0)) and
                    generated["detailed"]==show_text(response(facts,1))
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,default=0)
    p.add_argument("--kind",choices=("iid","ood"),default="ood")
    p.add_argument("--checkpoint",type=Path,default=DEFAULT_CKPT)
    p.add_argument("--all",action="store_true",help="Evaluate all heldout and summarize.")
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.all:
        rows=[run_example(i,a.kind,a.checkpoint) for i in range(48)]
        result={"kind":a.kind,"exact_pairs":sum(x["both_exact"] for x in rows),
                "total":len(rows),"examples":[rows[0],rows[1],rows[2]]}
    else:
        result=run_example(a.index,a.kind,a.checkpoint)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=="__main__":main()
