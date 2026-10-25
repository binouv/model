"""Local, isolated reproduction. Never publishes anything."""
import argparse,datetime,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from checkpoint_parts import restore
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('destination',type=Path);p.add_argument('--minutes',type=float,default=70);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();d=a.destination.resolve();d.mkdir(parents=True,exist_ok=False);runtime=d.with_name(d.name+'_runtime');runtime.mkdir(exist_ok=False)
 for sub in ['data','metrics','logs','checkpoints','verification','artifacts','source_cards']:(d/sub).mkdir()
 shutil.copytree(ROOT/'src',d/'src',ignore=shutil.ignore_patterns('__pycache__'))
 for sub in ['data','source_cards']:
  for f in (ROOT/sub).iterdir():
   if f.is_file():shutil.copy2(f,d/sub/f.name)
 for name in ['README.md','REGISTRATION.md','DATA_LICENSES.md','requirements.lock.txt']:shutil.copy2(ROOT/name,d/name)
 shutil.copy2(ROOT/'checkpoints/config.json',d/'checkpoints/config.json')
 for name in ['overlap_audit.json','reference_status.json']:shutil.copy2(ROOT/'metrics'/name,d/'metrics'/name)
 prior=ROOT.parent/'ext2_20261022/artifacts'
 restore(prior/'selected_model_parts',d/'checkpoints/input_EXT2.safetensors');restore(prior/'parent_parts',d/'checkpoints/input_EXT256.safetensors')
 cfg=json.loads((ROOT/'config.json').read_text());cfg.update(parent_path='checkpoints/input_EXT2.safetensors',original_EXT256_path='checkpoints/input_EXT256.safetensors',runtime_directory=str(runtime),training_stop_utc=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=a.minutes)).isoformat(),reproduction_of='ext3_20261025');(d/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 files=[d/'config.json',d/'REGISTRATION.md',d/'data/manifest.json',d/'data/tokenizer.json',d/'checkpoints/config.json']+list((d/'src').glob('*.py'));(d/'source_manifest.json').write_text(json.dumps({str(f.relative_to(d)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
 if a.prepare_only:print('Prepared',d);return
 for script in ['init_dev.py','train.py','verify_protocol.py','verify_resume.py','evaluate_final.py','make_report.py','package_run.py']:
  with (runtime/(script[:-3]+'.log')).open('x') as log:subprocess.run([sys.executable,str(d/'src'/script)],cwd=d,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('Local reproduction finished; no publication performed',d)
if __name__=='__main__':main()
