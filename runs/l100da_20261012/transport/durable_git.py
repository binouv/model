"""Non-force publication with a durable branch before updating a busy main."""
from __future__ import annotations
import re
import time
from typing import Callable


def publish_commit(git: Callable[..., str], paths: list[str], message: str,
                   run_id: str, sequence: int, max_attempts: int = 12) -> str:
    if not re.fullmatch(r'[0-9]+', run_id) or not 0 <= sequence < 1000:
        raise ValueError('Invalid publication identity')
    for path in paths:
        if not (path.startswith('runs/l100da_20261012/') or
                path.startswith('models/l100da_20261012_step0256/')):
            raise ValueError('Publication outside this experiment')
        if '..' in path.split('/'):
            raise ValueError('Invalid publication path')
    git('add', '-f', '--', *paths)
    if git('diff', '--cached', '--name-only'):
        git('commit', '-m', message + ' [skip ci]')
    original = git('rev-parse', 'HEAD')
    savepoint = f'flygraph/l100da-artifacts-{run_id}-{sequence:03d}'
    # Store large objects durably first. This new ref never overwrites a branch.
    existing = git('ls-remote', '--heads', 'origin', 'refs/heads/' + savepoint)
    if existing:
        if existing.split()[0] != original:
            raise RuntimeError('Savepoint already exists at another commit')
    else:
        git('push', 'origin', 'HEAD:refs/heads/' + savepoint)
    print('Artifact savepoint: ' + savepoint + ' ' + original, flush=True)
    for attempt in range(max_attempts):
        git('fetch', 'origin', '+refs/heads/main:refs/remotes/origin/main')
        # Merge, never reset or force-push; unrelated concurrent edits survive.
        git('merge', '--no-edit', 'refs/remotes/origin/main')
        try:
            git('push', 'origin', 'HEAD:refs/heads/main')
            return git('rev-parse', 'HEAD')
        except RuntimeError:
            if attempt + 1 == max_attempts:
                raise RuntimeError('Main publication failed; exact artifacts remain at ' + savepoint) from None
            print('Concurrent main update; retrying non-force merge/push.', flush=True)
            time.sleep(1)
    raise AssertionError('Unreachable')
