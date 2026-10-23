"""Create and run an isolated reproduction from the published selected data."""
import argparse,datetime,json,os,shutil,subprocess,sys,time
from pathlib import Path
from checkpoint_parts import restore
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);p.add_argument('--hours',type=float,default=4.);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();dst=a.destination.resolve();dst.mkdir(parents=True,exist_ok=False)
    for sub in ['data','checkpoints','metrics','logs','artifacts','verification','source_cards']:(dst/sub).mkdir()
    shutil.copytree(ROOT/'src',dst/'src',ignore=shutil.ignore_patterns('__pycache__'))
    for f in (ROOT/'data').iterdir():
        if f.is_file():shutil.copy2(f,dst/'data'/f.name)
    for name in ['requirements.lock.txt','REGISTRATION.md','DATA_LICENSES.md']:
        if (ROOT/name).exists():shutil.copy2(ROOT/name,dst/name)
    shutil.copy2(ROOT/'checkpoints/config.json',dst/'checkpoints/config.json');shutil.copy2(ROOT/'metrics/reference_status.json',dst/'metrics/reference_status.json')
    # The historical L100EC diagnosis is retained evidence, not recomputed from
    # a missing older ancestor. The new training/test pipeline is rerun below.
    shutil.copy2(ROOT/'metrics/diagnosis.json',dst/'metrics/diagnosis.json')
    for name in ['prior_training_overlap_audit.json','tokenizer_inventory_audit.json','frozen_run_audit.json']:
        if (ROOT/'metrics'/name).exists():shutil.copy2(ROOT/'metrics'/name,dst/'metrics'/name)
    for f in (ROOT/'source_cards').iterdir():
        if f.is_file():shutil.copy2(f,dst/'source_cards'/f.name)
    if (ROOT/'metrics/math_registration.json').exists():shutil.copy2(ROOT/'metrics/math_registration.json',dst/'metrics/math_registration.json')
    if (ROOT/'artifacts/parent_parts').exists():restore(ROOT/'artifacts/parent_parts',dst/'checkpoints/parent.safetensors')
    else:
        old=json.loads((ROOT/'config.json').read_text());shutil.copy2((ROOT/old['parent_path']).resolve(),dst/'checkpoints/parent.safetensors')
    cfg=json.loads((ROOT/'config.json').read_text());cfg['parent_path']='checkpoints/parent.safetensors';cfg['training_stop_utc']=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=a.hours)).isoformat();cfg['reproduction_of']='ext2_20261022';(dst/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print('Isolated reproduction prepared:',dst,flush=True)
    if a.prepare_only:return
    # Monitor files only; exactly one child performs training. Preserve step128
    # before the trainer prunes redundant recovery checkpoints.
    with (dst/'logs/train_stdout.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(dst/'src/train.py')],stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            source=dst/'checkpoints/step_0128/model.safetensors';target=dst/'verification/step_0128_model.safetensors'
            if source.exists() and not target.exists():os.link(source,target)
            time.sleep(5)
        if child.returncode:raise RuntimeError('Training failed; inspect preserved logs and checkpoint bytes.')
    for script in ['verify_protocol.py','verify_resume.py','evaluate_final.py','evaluate_math.py','make_report.py']:
        subprocess.run([sys.executable,str(dst/'src'/script)],check=True)
    print('Reproduction completed. It was not pushed or externally uploaded.',flush=True)
if __name__=='__main__':main()
