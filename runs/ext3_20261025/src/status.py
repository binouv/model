import json,datetime,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=ROOT/'PIPELINE_STATE.json';s=json.loads(p.read_text()) if p.exists() else {};log=(ROOT/'logs/train.jsonl').read_text().splitlines();last=json.loads(log[-1]);proc=Path('/proc/639/cmdline');alive=proc.exists() and 'ext3_20261025/src/train.py' in proc.read_bytes().decode(errors='replace')
 s.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),trainer_pid=639,training_process_alive=alive,completed_steps=last['step'],last_loss=last['loss'],last_log_mtime=datetime.datetime.fromtimestamp((ROOT/'logs/train.jsonl').stat().st_mtime,datetime.timezone.utc).isoformat(),training_finished=(ROOT/'metrics/training_summary.json').exists(),test_started=(ROOT/'metrics/final_evaluation_lock.json').exists())
 dev=sorted((ROOT/'metrics').glob('step_*_balanced_dev_summary.json'))
 if dev:s['latest_dev']=json.loads(dev[-1].read_text())
 if (ROOT/'metrics/selection.json').exists():s['selection']=json.loads((ROOT/'metrics/selection.json').read_text())
 tmp=ROOT/'STATUS.json.tmp';tmp.write_text(json.dumps(s,indent=2)+'\n');os.replace(tmp,ROOT/'STATUS.json');print(json.dumps(s))
if __name__=='__main__':main()
