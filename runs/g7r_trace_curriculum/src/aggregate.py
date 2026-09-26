"""Aggregate only completed fixed-step G7R endpoints."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARMS=("trace_then_final","final_only");SEEDS=(7801,7802,7803)
SPLITS=("test_iid","test_surface","test_extrapolation","test_counterfactual")

def sign_p(w,l):
    n=w+l
    if n==0:return 1.0
    k=min(w,l)
    tail=sum(math.comb(n,i) for i in range(k+1))/(2**n)
    return min(1.0,2*tail)

def main():
    rr={(a,s):json.loads((ROOT/"metrics"/f"{a}_s{s}.json").read_text()) for a in ARMS for s in SEEDS}
    assert all(z["status"]=="completed" and z["steps"]==2000 and z["reload_mismatches"]==0 and z["resume_next_update_verified"] for z in rr.values())
    assert len({rr[(a,s)]["batch_sequence_sha256"] for a in ARMS for s in SEEDS})==1
    results={}
    for arm in ARMS:
        results[arm]={}
        for sp in SPLITS:
            vals=[rr[(arm,s)]["tests"][sp] for s in SEEDS]
            results[arm][sp]={
                "mean_accuracy":sum(v["accuracy"] for v in vals)/3,
                "correct_by_seed":[v["correct"] for v in vals],
                "n_unique":vals[0]["n"],
                "both_correct_pairs_by_seed":[v["both_correct_pairs"] for v in vals],
                "pair_n":vals[0]["pair_count"]
            }
        results[arm]["trace_diagnostic_iid"]={
            "mean_exact_trace":sum(rr[(arm,s)]["trace_diagnostic_iid"]["accuracy"] for s in SEEDS)/3,
            "exact_trace_by_seed":[rr[(arm,s)]["trace_diagnostic_iid"]["exact_trace"] for s in SEEDS],
            "n_unique":rr[(arm,SEEDS[0])]["trace_diagnostic_iid"]["n"]
        }
        results[arm]["train"]={
            "input_tokens_by_seed":[rr[(arm,s)]["input_tokens"] for s in SEEDS],
            "supervised_tokens_by_seed":[rr[(arm,s)]["supervised_tokens"] for s in SEEDS],
            "train_seconds_by_seed":[rr[(arm,s)]["train_seconds"] for s in SEEDS],
            "profiler_supported_flops_one_batch_by_seed":[rr[(arm,s)]["profiler"]["supported_forward_backward_flops"] for s in SEEDS]
        }
    paired={}
    for seed in SEEDS:
        paired[str(seed)]={}
        for sp in SPLITS:
            A={x["id"]:x for x in rr[("trace_then_final",seed)]["tests"][sp]["rows"]}
            B={x["id"]:x for x in rr[("final_only",seed)]["tests"][sp]["rows"]}
            assert set(A)==set(B)
            w=sum(A[k]["exact"] and not B[k]["exact"] for k in A);l=sum(B[k]["exact"] and not A[k]["exact"] for k in A)
            paired[str(seed)][sp]={"wins":w,"losses":l,"delta_pp":100*(sum(x["exact"] for x in A.values())-sum(x["exact"] for x in B.values()))/len(A),"exact_two_sided_p":sign_p(w,l)}
    gains={sp:100*(results["trace_then_final"][sp]["mean_accuracy"]-results["final_only"][sp]["mean_accuracy"]) for sp in SPLITS}
    pair_main=sum(results["trace_then_final"]["test_counterfactual"]["both_correct_pairs_by_seed"])/(3*results["trace_then_final"]["test_counterfactual"]["pair_n"])
    pair_ctrl=sum(results["final_only"]["test_counterfactual"]["both_correct_pairs_by_seed"])/(3*results["final_only"]["test_counterfactual"]["pair_n"])
    gates={
      "iid_ge5pp":gains["test_iid"]>=5,
      "surface_extrapolation_pooled_positive":gains["test_surface"]+gains["test_extrapolation"]>0,
      "counterfactual_pair_regression_le2pp":100*(pair_main-pair_ctrl)>=-2,
      "positive_iid_at_least2seeds":sum(paired[str(s)]["test_iid"]["delta_pp"]>0 for s in SEEDS)>=2
    }
    family={}
    fams=sorted(rr[("trace_then_final",SEEDS[0])]["family_iid"])
    for f in fams:
        family[f]={a:{"correct_by_seed":[rr[(a,s)]["family_iid"][f]["correct"] for s in SEEDS],
                      "n_per_seed":rr[(a,SEEDS[0])]["family_iid"][f]["n"]} for a in ARMS}
    out={"run":"G7R","status":"completed","training_models":6,"seeds":list(SEEDS),"parameters_each":469648,"steps_each":2000,
         "results":results,"paired_exact":paired,"gains_pp":gains,
         "counterfactual_both_correct_pair_gain_pp":100*(pair_main-pair_ctrl),"gates":gates,
         "verdict":"CONFIRMED_IN_REGISTERED_PROTOCOL" if all(gates.values()) else "NOT_CONFIRMED_IN_REGISTERED_PROTOCOL",
         "family_iid":family,"same_held_cases_across_seeds":True,"same_batch_sequence_across_all_models":True,
         "scope":"469,648-parameter English synthetic mechanism probe; not 300-800M, not broad code generation, not Qwen parity"}
    (ROOT/"metrics/FINAL.json").write_text(json.dumps(out,indent=2))
    with (ROOT/"metrics/FINAL.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["split","trace_then_final","final_only","gain_pp"])
        for sp in SPLITS:w.writerow([sp,results["trace_then_final"][sp]["mean_accuracy"],results["final_only"][sp]["mean_accuracy"],gains[sp]])
        w.writerow(["trace_diagnostic_iid",results["trace_then_final"]["trace_diagnostic_iid"]["mean_exact_trace"],results["final_only"]["trace_diagnostic_iid"]["mean_exact_trace"],100*(results["trace_then_final"]["trace_diagnostic_iid"]["mean_exact_trace"]-results["final_only"]["trace_diagnostic_iid"]["mean_exact_trace"])])
    print(json.dumps({"verdict":out["verdict"],"gains_pp":gains,"gates":gates}))
if __name__=="__main__":main()
