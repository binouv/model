"""Replay the frozen sampler without training and audit target-token weighting."""
from __future__ import annotations
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit() -> dict:
    config = json.loads((ROOT / 'configs/preregistered.json').read_text())
    rows = [json.loads(line) for line in (ROOT / 'data/train.jsonl').read_text().splitlines()]
    logs = [json.loads(line) for line in (ROOT / 'metrics/train.jsonl').read_text().splitlines()]
    pools = {(family, level): [row for row in rows if row['family'] == family and row['level'] == level]
             for family in config['families'] for level in ('elementary', 'iid')}
    rng = random.Random(config['sampler_seed'])
    examples, tokens = Counter(), Counter()
    if len(logs) != config['updates']:
        raise ValueError('Run is not the complete fixed endpoint')
    for step, log in enumerate(logs, 1):
        picked = []
        for j in range(config['batch']):
            family = config['families'][((step - 1) * config['batch'] + j) % len(config['families'])]
            level = 'elementary' if step <= config['elementary_updates'] or rng.random() < .25 else 'iid'
            picked.append(rng.choice(pools[family, level]))
        picked.sort(key=lambda row: len(row['tokens']))
        observed = hashlib.sha256('|'.join(row['id'] for row in picked).encode()).hexdigest()
        if log['step'] != step or log['batch_ids_sha256'] != observed:
            raise ValueError(f'Sampler mismatch at optimizer step {step}')
        count = sum(len(row['tokens']) - row['prompt_tokens'] for row in picked)
        if count != log['supervised_tokens']:
            raise ValueError('Target mask count differs from actual training log')
        for row in picked:
            examples[row['family']] += 1
            tokens[row['family']] += len(row['tokens']) - row['prompt_tokens']
    total = sum(tokens.values())
    return {
        'status': 'completed_retrospective_diagnostic',
        'candidate_optimizer_updates_added': 0,
        'sampler_batches_replayed_and_hash_matched': len(logs),
        'total_supervised_tokens': total,
        'families': {f: {'presentations': examples[f], 'supervised_tokens': tokens[f],
                          'supervised_token_fraction': tokens[f] / total} for f in config['families']},
        'interpretation': 'The fixed objective averages across supervised tokens, not examples. Long code answers contribute more token terms. This is an observed weighting imbalance, NOT a causal demonstration of why numeric transfer failed.',
        'next_testable_hypothesis': 'Compare per-example-normalized answer loss against current token-normalized loss with a matched checkpoint, sampler and fixed training budget; no claim of Qwen superiority.'
    }

if __name__ == '__main__':
    result = audit()
    previous = ROOT / 'diagnostics/LOSS_BUDGET.json'
    if previous.exists() and json.loads(previous.read_text()) != result:
        raise ValueError('Audit differs from the preserved diagnostic')
    print(json.dumps(result, indent=2))
