"""CLI for a saved synthetic latent-plan checkpoint. Trust checkpoint source."""
import argparse
from pathlib import Path
import torch
from latent_answer import AGENTS, VERBS, OBJECTS, LatentSystem, readable

@torch.inference_mode()
def generate(checkpoint: str, agent: str, verb: str, object_: str, style: str = 'both'):
    data = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
    net = LatentSystem().eval()
    net.planner.load_state_dict(data['planner'], strict=True)
    net.renderer.load_state_dict(data['renderer'], strict=True)
    index = (AGENTS.index(agent), VERBS.index(verb), OBJECTS.index(object_))
    plan = net.planner(torch.tensor([index], dtype=torch.long))
    output = {}
    selected = [0, 1] if style == 'both' else [0 if style == 'short' else 1]
    for style_index in selected:
        tokens = net.renderer.generate(plan, torch.tensor([style_index]))[0].tolist()
        output['short' if style_index == 0 else 'detailed'] = readable(tokens)
    return {'facts': index, 'latent_dimensions': int(plan.shape[-1]), 'planner_calls': 1, 'outputs': output}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='results/latent_answer_training_checkpoint.pt')
    p.add_argument('--agent', choices=AGENTS, default='owl')
    p.add_argument('--verb', choices=VERBS, default='finds')
    p.add_argument('--object', dest='object_', choices=OBJECTS, default='map')
    p.add_argument('--style', choices=('short', 'detailed', 'both'), default='both')
    a = p.parse_args()
    print(generate(a.checkpoint, a.agent, a.verb, a.object_, a.style))
