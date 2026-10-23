"""Refresh volatile status from real process/log files without launching compute."""
import datetime,json,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=ROOT/'PIPELINE_STATE.json';s=json.loads(p.read_text()) if p.exists() else {}
    log=(ROOT/'logs/train.jsonl').read_text().splitlines();t=json.loads(log[-1])
    proc=Path('/proc/765/cmdline');alive=proc.exists() and 'ext2_20261022/src/train.py' in proc.read_bytes().decode(errors='replace')
    s.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),existing_pid=765,training_process_alive=alive,new_training_process_started=False,completed_steps=t['step'],latest_loss=t['loss'],training_finished=(ROOT/'metrics/training_summary.json').exists(),test_started=(ROOT/'metrics/final_evaluation_lock.json').exists(),verified_checkpoint_directories=sorted(p.name for p in (ROOT/'checkpoints').glob('step_[0-9][0-9][0-9][0-9]') if (p/'manifest.json').exists()))
    dev=sorted((ROOT/'metrics').glob('step_*_dev_summary.json'))
    if dev:
        d=json.loads(dev[-1].read_text());s['latest_dev']={'arm':d['arm'],'alpaca_nll':d['alpaca_nll'],'selection_objective':d['selection_objective'],'by_task':{k:{'correct':v['label_correct'],'n':v['n']} for k,v in d['by_task'].items()}}
    s['selection']=json.loads((ROOT/'metrics/selection.json').read_text())
    temp=ROOT/'STATUS.json.tmp';temp.write_text(json.dumps(s,indent=2)+'\n');os.replace(temp,ROOT/'STATUS.json')
    print(json.dumps(s))
if __name__=='__main__':main()
