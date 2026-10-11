"""Resume transfer using durable Git savepoints; never replay model training."""
from __future__ import annotations
import importlib.util
import os
import re
import sys
import traceback
from pathlib import Path
from durable_git import publish_commit

spec = importlib.util.spec_from_file_location('resume_transfer', Path(__file__).with_name('resumable.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
sequence = 0


def commit(paths: list[str], message: str) -> str:
    global sequence
    sequence += 1
    return publish_commit(r.p.cp.git, paths, message, os.environ['GITHUB_RUN_ID'], sequence)


if __name__ == '__main__':
    r.p.commit = commit
    try:
        r.run()
    except Exception as error:
        # Log the actual stage without any signed artifact URLs or credentials.
        text = str(error)
        for secret in (os.environ.get('GH_TOKEN'), os.environ.get('GIT_CONFIG_VALUE_0')):
            if secret:
                text = text.replace(secret, '[REDACTED]')
        text = re.sub(r'https?://[^\s]+', '[URL]', text)
        print(type(error).__name__ + ': ' + text[:4000], file=sys.stderr)
        for frame in traceback.extract_tb(error.__traceback__):
            print(f'{Path(frame.filename).name}:{frame.lineno}:{frame.name}', file=sys.stderr)
        raise SystemExit(1) from None
