"""Wait for the existing trainer, then run a single bounded completion pipeline."""
import datetime,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,atomic_json

PID=765
RUNTIME=Path('/workspace/ext2_runtime_20261022')
DEADLINE=datetime.datetime.fromisoformat('2026-10-23T02:25:00+00:00').timestamp()

def main():
    RUNTIME.mkdir(exist_ok=True)
    with (ROOT/'PIPELINE_LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            proc=Path(f'/proc/{PID}/cmdline')
            alive=proc.exists() and 'ext2_20261022/src/train.py' in proc.read_bytes().decode(errors='replace')
            if not alive:break
            atomic_json(ROOT/'PIPELINE_STATE.json',{'stage':'waiting_for_existing_trainer','trainer_pid':PID,'controller_pid':os.getpid(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
            time.sleep(20)
        assert (ROOT/'metrics/training_summary.json').exists(),'Trainer stopped without completion; inspect checkpoint before any recovery'
        stages=[('verify_protocol.py','metrics/protocol_verification.json'),('verify_resume.py','metrics/resume_verification.json'),('evaluate_final.py','metrics/final_results.json'),('evaluate_math.py','metrics/math_results.json'),('make_report.py','REPORT_RU.md'),('package_run.py','artifacts/MANIFEST.json'),('publish_completed.py','PUSH_COMPLETE.json')]
        for script,output in stages:
            if time.time()>DEADLINE:raise TimeoutError('Completion pipeline cutoff reached; preserve results for manual inspection')
            # The controller is started only once, but existing results must be
            # inspected rather than silently rerun or overwritten.
            assert not (ROOT/output).exists(),f'Existing stage output needs inspection: {output}'
            atomic_json(ROOT/'PIPELINE_STATE.json',{'stage':script,'controller_pid':os.getpid(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
            print('STAGE',script,flush=True)
            logfile=RUNTIME/(script.removesuffix('.py')+'.log')
            with logfile.open('x') as log:
                subprocess.run([sys.executable,str(ROOT/'src'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert (ROOT/output).exists(),f'Missing output from {script}'
        atomic_json(ROOT/'PIPELINE_STATE.json',{'stage':'completed','controller_pid':os.getpid(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        print('PIPELINE_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        atomic_json(ROOT/'PIPELINE_STATE.json',{'stage':'failed_stopped_no_retry','error':repr(e),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise
