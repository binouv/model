from __future__ import annotations
import csv, json, math, time
from pathlib import Path
import numpy as np
import r35_base as r

# FlyGraph R36: trajectory-verifier uncertainty as an adaptive compute allocator.
# Uses the frozen R35 verifier; no new learned parameters are introduced.

R36_CONDS = {
    **r.CONDS,
    'ood256': (256, 576, 1024, .50),
}


def load_base(asset_dir: Path):
    w = np.load(asset_dir / 'r35_verifier_weights.npz', allow_pickle=True)
    model = {
        'cols': w['cols'], 'mean': w['mean'], 'scale': w['scale'],
        'coef': w['coef'], 'intercept': float(w['intercept'][0])
    }
    alpha = float(w['local_alpha'][0])
    rng = np.random.default_rng(3000)
    perms = np.stack([rng.permutation(r.S) for _ in range(r.S)])
    return model, alpha, perms


def fixed_verifier(ep, perms, model, alpha, width=8, expand=8, prepared=None):
    """R35 scorer with instrumentation. Inference ranking never uses oracle labels."""
    sids, latest, fut_tab, fm_tab = prepared if prepared is not None else r.prep(ep)
    hyps = [(0., ep['q'], int(ep['start']), True, frozenset(), 0., 0., 0.)]
    surv = []
    verifier_evals = 0
    for j, rel in enumerate(ep['prog']):
        cand = []
        for score, q, st, pok, used, clp, cm, cf in hyps:
            logits = 10 * (ep['sub'] @ q) + 3 * (ep['rel'] == rel) + 2 * ep['ts']
            p, order, margin, ent = r.softstats(logits)
            for ix0 in order[:expand]:
                verifier_evals += 1
                ix = int(ix0)
                f = r.feat(ep, j, q, ix, p, margin, ent, sids, latest, fut_tab, fm_tab,
                           used, clp, cm, cf)
                ns = score + r.verifier_score(model, f) + alpha * f[3]
                key = (int(sids[ix]), int(ep['rel'][ix]))
                cand.append((ns, ep['obj'][ix], int(perms[int(ep['op'][ix]), st]),
                             pok and ix == int(ep['idx'][j]), used | {key},
                             clp + f[3], cm + margin, cf + f[8]))
        cand.sort(key=lambda x: x[0], reverse=True)
        hyps = cand[:width]
        surv.append(any(h[3] for h in hyps))
    best = hyps[0]
    return {
        'state': int(best[2]),
        'path_survival': float(np.mean(surv)),
        'final_true_path_in_beam': float(surv[-1]),
        'selected_true_path': float(best[3]),
        'verifier_candidate_evals': int(verifier_evals),
        'triggered': False,
        'trigger_step': None,
    }


def adaptive_verifier(ep, perms, model, alpha, threshold=.10, signal='verifier',
                      base_width=8, max_width=32, expand=8, prepared=None):
    """
    Sticky adaptive beam. Start at width 8. At a low-confidence step widen to 32
    before pruning and keep that width for the remainder of the trajectory.

    signal='verifier': top1-top2 *trajectory verifier* candidate-score gap (R36 main).
    signal='local': top1-top2 raw retrieval-logit gap from the current best prefix
                    (single confirming ablation).

    Oracle idx/pok is tracked only for evaluation metrics; it never enters the ranking score.
    """
    sids, latest, fut_tab, fm_tab = prepared if prepared is not None else r.prep(ep)
    hyps = [(0., ep['q'], int(ep['start']), True, frozenset(), 0., 0., 0.)]
    surv = []
    width = base_width
    verifier_evals = 0
    trigger_step = None

    for j, rel in enumerate(ep['prog']):
        # Ablation's uncertainty signal: local retrieval margin from current best prefix.
        q0 = hyps[0][1]
        logits0 = 10 * (ep['sub'] @ q0) + 3 * (ep['rel'] == rel) + 2 * ep['ts']
        order0 = np.argsort(logits0)[::-1]
        local_gap = float(logits0[order0[0]] - logits0[order0[1]]) if len(order0) > 1 else 99.0

        cand = []
        for score, q, st, pok, used, clp, cm, cf in hyps:
            logits = 10 * (ep['sub'] @ q) + 3 * (ep['rel'] == rel) + 2 * ep['ts']
            p, order, margin, ent = r.softstats(logits)
            for ix0 in order[:expand]:
                verifier_evals += 1
                ix = int(ix0)
                f = r.feat(ep, j, q, ix, p, margin, ent, sids, latest, fut_tab, fm_tab,
                           used, clp, cm, cf)
                ns = score + r.verifier_score(model, f) + alpha * f[3]
                key = (int(sids[ix]), int(ep['rel'][ix]))
                cand.append((ns, ep['obj'][ix], int(perms[int(ep['op'][ix]), st]),
                             pok and ix == int(ep['idx'][j]), used | {key},
                             clp + f[3], cm + margin, cf + f[8]))

        cand.sort(key=lambda x: x[0], reverse=True)
        verifier_gap = float(cand[0][0] - cand[1][0]) if len(cand) > 1 else 99.0
        gap = verifier_gap if signal == 'verifier' else local_gap
        if width == base_width and gap < threshold:
            width = max_width
            trigger_step = j

        hyps = cand[:width]
        surv.append(any(h[3] for h in hyps))

    best = hyps[0]
    return {
        'state': int(best[2]),
        'path_survival': float(np.mean(surv)),
        'final_true_path_in_beam': float(surv[-1]),
        'selected_true_path': float(best[3]),
        'verifier_candidate_evals': int(verifier_evals),
        'triggered': trigger_step is not None,
        'trigger_step': int(trigger_step) if trigger_step is not None else None,
    }


def greedy_instrumented(ep, perms):
    st, addr, full = r.greedy(ep, perms)
    return {
        'state': int(st), 'path_survival': float(addr),
        'final_true_path_in_beam': float(full), 'selected_true_path': float(full),
        'verifier_candidate_evals': 0, 'triggered': False, 'trigger_step': None,
        'address_step_acc': float(addr),
    }


def summarize_rows(rows):
    out = {}
    scalar_keys = ['answer', 'path_survival', 'final_true_path_in_beam',
                   'selected_true_path', 'verifier_candidate_evals']
    for k in scalar_keys:
        vals = np.asarray([x[k] for x in rows], dtype=np.float64)
        out[k] = {'mean': float(vals.mean()), 'std': float(vals.std()),
                  'values': [float(v) for v in vals]}
    trig = np.asarray([x['triggered'] for x in rows], dtype=np.float64)
    out['trigger_rate'] = float(trig.mean())
    steps = [x['trigger_step'] for x in rows if x['trigger_step'] is not None]
    out['trigger_step_mean_when_triggered'] = float(np.mean(steps)) if steps else None
    if any('address_step_acc' in x for x in rows):
        vals = [x['address_step_acc'] for x in rows if 'address_step_acc' in x]
        out['address_step_acc'] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                                   'values': [float(v) for v in vals]}
    else:
        out['address_step_acc'] = None
    return out



def run_one(method, ep, prepared, perms, model, alpha, threshold):
    if method == 'greedy':
        z = greedy_instrumented(ep, perms)
    elif method == 'r35_fixed8':
        z = fixed_verifier(ep, perms, model, alpha, width=8, expand=8, prepared=prepared)
    elif method == 'fixed32':
        z = fixed_verifier(ep, perms, model, alpha, width=32, expand=8, prepared=prepared)
    elif method == 'r36_adaptive_verifier':
        z = adaptive_verifier(ep, perms, model, alpha, threshold=threshold, signal='verifier', prepared=prepared)
    elif method == 'ablation_local_margin':
        z = adaptive_verifier(ep, perms, model, alpha, threshold=threshold, signal='local', prepared=prepared)
    else:
        raise ValueError(method)
    z['answer'] = float(z['state'] == int(ep['target']))
    return z


def summarize_grouped(rows):
    summary = summarize_rows(rows)
    by_seed = []
    for sd in sorted(set(int(x['seed']) for x in rows)):
        sr = [x for x in rows if int(x['seed']) == sd]
        by_seed.append({'seed': sd, **summarize_rows(sr)})
    summary['seeds'] = by_seed
    summary['n_episodes'] = len(rows)
    return summary


def eval_condition(args, seeds, n_per_seed, methods, perms, model, alpha, threshold):
    rows = {m: [] for m in methods}
    for sd in seeds:
        for i in range(n_per_seed):
            ep = r.epgen(*args, .30, sd * 100000 + i, perms)
            prepared = r.prep(ep)
            for method in methods:
                z = run_one(method, ep, prepared, perms, model, alpha, threshold)
                z['seed'] = int(sd); z['episode'] = int(i)
                rows[method].append(z)
    return {m: summarize_grouped(v) for m, v in rows.items()}


def calibration_episodes(perms, cal_seeds, n_cal):
    cache = {}
    for cn in ('id8', 'ood64', 'ood128'):
        items = []
        for sd in cal_seeds:
            for i in range(n_cal):
                ep = r.epgen(*R36_CONDS[cn], .30, sd * 100000 + i, perms)
                items.append((sd, i, ep, r.prep(ep)))
        cache[cn] = items
    return cache


def eval_cached(items, method, perms, model, alpha, threshold):
    rows=[]
    for sd,i,ep,prepared in items:
        z=run_one(method,ep,prepared,perms,model,alpha,threshold)
        z['seed']=int(sd); z['episode']=int(i); rows.append(z)
    return summarize_grouped(rows)


def calibrate(perms, model, alpha, thresholds, cal_seeds, n_cal):
    cache = calibration_episodes(perms, cal_seeds, n_cal)
    refs = {cn: eval_cached(cache[cn], 'fixed32', perms, model, alpha, .1)
            for cn in ('id8','ood64','ood128')}
    rows=[]
    for th in thresholds:
        vals={cn: eval_cached(cache[cn], 'r36_adaptive_verifier', perms, model, alpha, th)
              for cn in ('id8','ood64','ood128')}
        cr=float(np.mean([vals[cn]['verifier_candidate_evals']['mean']/refs[cn]['verifier_candidate_evals']['mean']
                          for cn in ('id8','ood64','ood128')]))
        obj=(2*vals['id8']['answer']['mean'] + vals['ood64']['answer']['mean']
             + 2*vals['ood128']['answer']['mean']
             + .5*vals['ood128']['final_true_path_in_beam']['mean'] - .10*cr)
        rows.append({'threshold':float(th),'objective':float(obj),'mean_compute_ratio_vs_fixed32':cr,'metrics':vals})
    best=max(rows,key=lambda x:(x['objective'],-x['threshold']))
    return rows,float(best['threshold'])


def make_report(payload):
    H=payload['held_results']; E=payload['extension_256']; v=payload['verdict']; th=payload['selected_threshold']
    def pct(x): return f'{100*x:.2f}%'
    def ev(x): return f'{x:.0f}'
    L=[
        '# FlyGraph R36 — verifier-uncertainty adaptive beam allocation','',
        '**Статус:** completed  ','**Приоритет:** cognition — memory/reasoning; audio/video не затрагивались.','',
        '## Точная гипотеза','',
        'Неопределённость уже подтверждённого R35 trajectory verifier можно использовать как управляющий сигнал вычислительного бюджета: начинать с beam width 8 и sticky-расширяться до width 32 только при малом top1–top2 verifier trajectory-score gap. Это должно улучшить 128-hop noisy-memory reasoning относительно R35 fixed8, сохранить 8-hop accuracy в пределах 1–2 п.п. от greedy и тратить меньше verifier-candidate evaluations, чем always-on width32.','',
        f'Порог выбран только на calibration seeds: `{th}`. R35 verifier weights frozen; новых обучаемых параметров нет.','',
        '## Протокол','',
        f"- Noise: bit-flip `p={payload['protocol']['noise']}`.",
        f"- Calibration seeds: {payload['protocol']['calibration_seeds']}; held seeds: {payload['protocol']['held_seeds']}.",
        f"- Held N: {payload['protocol']['n_held_by_condition']} на seed; 256 extension: {payload['protocol']['n_256_per_seed']} на seed.",
        '- Main: verifier trajectory-gap trigger, width 8→32 sticky.',
        '- Единственная абляция: тот же policy и тот же threshold, но trigger по raw local retrieval margin.','',
        '## Held-out answer accuracy','',
        '| Hops | N | Greedy | R35 fixed8 | Fixed32 | **R36 adaptive** | Local-margin ablation |','|---:|---:|---:|---:|---:|---:|---:|'
    ]
    for cn,h in [('id8',8),('ood32',32),('ood64',64),('ood128',128)]:
        n=H[cn]['r36_adaptive_verifier']['n_episodes']
        L.append(f"| {h} | {n} | {pct(H[cn]['greedy']['answer']['mean'])} | {pct(H[cn]['r35_fixed8']['answer']['mean'])} | {pct(H[cn]['fixed32']['answer']['mean'])} | **{pct(H[cn]['r36_adaptive_verifier']['answer']['mean'])}** | {pct(H[cn]['ablation_local_margin']['answer']['mean'])} |")
    L += ['', '## Memory / reasoning metrics','',
          '| Hops | Method | Path survival | Final true path | Selected exact path | Verifier evals | Trigger rate | Trigger step* |','|---:|---|---:|---:|---:|---:|---:|---:|']
    for cn,h in [('id8',8),('ood32',32),('ood64',64),('ood128',128)]:
        for m,label in [('r35_fixed8','R35 fixed8'),('fixed32','Fixed32'),('r36_adaptive_verifier','R36 adaptive'),('ablation_local_margin','Local-margin ablation')]:
            z=H[cn][m]; ts=z['trigger_step_mean_when_triggered']
            L.append(f"| {h} | {label} | {pct(z['path_survival']['mean'])} | {pct(z['final_true_path_in_beam']['mean'])} | {pct(z['selected_true_path']['mean'])} | {ev(z['verifier_candidate_evals']['mean'])} | {pct(z['trigger_rate'])} | {('—' if ts is None else f'{ts:.1f}')} |")
    L += ['', '*Trigger step считается с нуля и усредняется только по сработавшим эпизодам.','',
          '## 256-hop extension','',
          f"Завершено {E['r36_adaptive_verifier']['n_episodes']} held-out эпизодов.", '',
          '| Method | Answer | Path survival | Final true path | Selected exact path | Verifier evals |','|---|---:|---:|---:|---:|---:|']
    for m,label in [('greedy','Greedy'),('r35_fixed8','R35 fixed8'),('fixed32','Fixed32'),('r36_adaptive_verifier','R36 adaptive'),('ablation_local_margin','Local-margin ablation')]:
        z=E[m]; L.append(f"| {label} | {pct(z['answer']['mean'])} | {pct(z['path_survival']['mean'])} | {pct(z['final_true_path_in_beam']['mean'])} | {pct(z['selected_true_path']['mean'])} | {ev(z['verifier_candidate_evals']['mean'])} |")
    L += ['', '## Вердикт','', f"**{v['status']}** — {v['summary']}", '',
          f"8-hop regression vs greedy: {v['short_regression_pp']:+.2f} п.п.; 128-hop gain vs R35 fixed8: {v['gain_128_vs_fixed8_pp']:+.2f} п.п.; verifier-eval saving vs fixed32 at 128: {v['compute_saving_128_pct']:.2f}%.", '',
          '## Следующий шаг','', payload['next_step'], '',
          '## Артефакты','',
          '`r36_metrics.json`, `r36_metrics.csv`, `r36_adaptive_verifier_beam.py`, `r36_config.json`, `test_r36_semantics.py`; frozen dependency snapshot: `r35_base.py`, `r35_verifier_weights.npz`.']
    return '\n'.join(L)+'\n'


def main(outdir):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); t0=time.time()
    model,alpha,perms=load_base(outdir)
    config={
        'experiment':'FlyGraph R36 verifier-uncertainty adaptive beam allocation','noise':.30,
        'calibration_seeds':[3691,3692],'held_seeds':[3701,3702,3703],
        'n_cal_per_seed':4,'n_held_by_condition':{'id8':16,'ood32':12,'ood64':12,'ood128':8},
        'n_256_per_seed':4,'threshold_grid':[.05,.10,.20],
        'base_width':8,'max_width':32,'expand':8,'r35_local_alpha':alpha,
        'success_criterion':'Improve held 128-hop answer vs R35 fixed8; 8-hop regression >= -2pp vs greedy; verifier-candidate evals at 128 lower than fixed32.'
    }
    (outdir/'r36_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    calibration,threshold=calibrate(perms,model,alpha,config['threshold_grid'],config['calibration_seeds'],config['n_cal_per_seed'])
    methods=['greedy','r35_fixed8','fixed32','r36_adaptive_verifier','ablation_local_margin']
    held={}
    for cn in ('id8','ood32','ood64','ood128'):
        held[cn]=eval_condition(R36_CONDS[cn],config['held_seeds'],config['n_held_by_condition'][cn],methods,perms,model,alpha,threshold)
    ext=eval_condition(R36_CONDS['ood256'],config['held_seeds'],config['n_256_per_seed'],methods,perms,model,alpha,threshold)
    short_reg=100*(held['id8']['r36_adaptive_verifier']['answer']['mean']-held['id8']['greedy']['answer']['mean'])
    gain128=100*(held['ood128']['r36_adaptive_verifier']['answer']['mean']-held['ood128']['r35_fixed8']['answer']['mean'])
    f32=held['ood128']['fixed32']['verifier_candidate_evals']['mean']; ad=held['ood128']['r36_adaptive_verifier']['verifier_candidate_evals']['mean']; saving=100*(1-ad/f32)
    success=(gain128>0 and short_reg>=-2 and saving>0)
    if success:
        status='CONFIRMED'; summary='verifier trajectory uncertainty is useful for adaptive compute allocation under the predeclared criterion.'
        next_step=('R37: fix the remaining trajectory-ranking failure rather than scale parameters. Train an off-policy prefix/trajectory verifier on beam-generated hard negatives, then reuse this allocator. Target: materially raise 128-hop exact-path selection and obtain non-zero exact-path survival at 256 before 100M/200M/300M workspace scaling.')
    else:
        status='REFUTED / MIXED'; summary='adaptive width alone did not satisfy all predeclared requirements; extra beam compute is not yet the solution.'
        next_step=('R37: keep protected/discrete memory and R35 consistency features, but train an off-policy prefix/trajectory verifier on beam-generated hard negatives. Do not scale parameter count until 128-hop exact-path ranking improves.')
    payload={'experiment':config['experiment'],'status':'completed',
             'hypothesis':'Trajectory-verifier uncertainty can allocate beam width adaptively: width 8→32 only after a low top1-top2 trajectory verifier margin should improve 128-hop noisy-memory reasoning over R35 fixed8, preserve 8-hop accuracy within 1-2pp of greedy, and use fewer verifier-candidate evaluations than always-on width32.',
             'protocol':{'noise':config['noise'],'calibration_seeds':config['calibration_seeds'],'held_seeds':config['held_seeds'],'n_cal_per_seed':config['n_cal_per_seed'],'n_held_by_condition':config['n_held_by_condition'],'n_256_per_seed':config['n_256_per_seed'],'base_width':8,'max_width':32,'expand':8,'r35_weights_frozen':True,'ablation':'same policy/threshold, raw local retrieval margin trigger'},
             'calibration':calibration,'selected_threshold':threshold,'held_results':held,'extension_256':ext,
             'verdict':{'status':status,'summary':summary,'short_regression_pp':short_reg,'gain_128_vs_fixed8_pp':gain128,'compute_saving_128_pct':saving,'criterion_met':bool(success)},
             'next_step':next_step,'elapsed_sec':time.time()-t0}
    (outdir/'r36_metrics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with (outdir/'r36_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        cw=csv.writer(f); cw.writerow(['condition','method','n','answer_mean','answer_std','path_survival_mean','final_true_path_in_beam','selected_true_path','verifier_candidate_evals_mean','trigger_rate','trigger_step_mean_when_triggered'])
        for cn,md in held.items():
            for m,z in md.items(): cw.writerow([cn,m,z['n_episodes'],z['answer']['mean'],z['answer']['std'],z['path_survival']['mean'],z['final_true_path_in_beam']['mean'],z['selected_true_path']['mean'],z['verifier_candidate_evals']['mean'],z['trigger_rate'],z['trigger_step_mean_when_triggered']])
        for m,z in ext.items(): cw.writerow(['ood256',m,z['n_episodes'],z['answer']['mean'],z['answer']['std'],z['path_survival']['mean'],z['final_true_path_in_beam']['mean'],z['selected_true_path']['mean'],z['verifier_candidate_evals']['mean'],z['trigger_rate'],z['trigger_step_mean_when_triggered']])
    (outdir/'RESEARCH_REPORT_R36_RU.md').write_text(make_report(payload),encoding='utf-8')
    print(json.dumps({'selected_threshold':threshold,'held_answer':{cn:{m:round(z['answer']['mean'],4) for m,z in md.items()} for cn,md in held.items()},'held_evals_128':{m:round(z['verifier_candidate_evals']['mean'],1) for m,z in held['ood128'].items()},'extension_256_answer':{m:round(z['answer']['mean'],4) for m,z in ext.items()},'verdict':payload['verdict'],'elapsed_sec':payload['elapsed_sec']},indent=2))


if __name__=='__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else str(Path(__file__).resolve().parent))