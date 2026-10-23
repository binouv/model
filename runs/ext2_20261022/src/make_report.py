"""Report only completed measurements and explicit missing references."""
import csv,json,statistics
from pathlib import Path
import numpy as np
from common import ROOT,atomic_json,sha
def main():
    from audit_results import main as audit_results
    tie_audit=audit_results()
    results=json.loads((ROOT/'metrics/final_results.json').read_text());assert results['evaluation_completed']
    train=[json.loads(s) for s in (ROOT/'logs/train.jsonl').read_text().splitlines()];diag=json.loads((ROOT/'metrics/diagnosis.json').read_text());sel=json.loads((ROOT/'metrics/selection.json').read_text());res=json.loads((ROOT/'metrics/resume_verification.json').read_text());assert res['all_135_native_tensors_bitwise_equal']
    assert [r['step'] for r in train]==list(range(1,results['training']['steps']+1))
    stats={'completed_updates':len(train),'input_tokens':sum(r['input_tokens'] for r in train),'supervised_tokens':sum(r['supervised_tokens'] for r in train),'unique_training_records':len({i for r in train for i in r['sample_ids']}),'first64_mean_example_loss':statistics.mean(r['loss'] for r in train[:64]),'last64_mean_example_loss':statistics.mean(r['loss'] for r in train[-64:]),'parent_sha256':json.loads((ROOT/'config.json').read_text())['parent_sha256'],'selected_step':sel['best_step'],'selected_sha256':results['summaries']['selected_EXT2']['weight_sha256']}
    atomic_json(ROOT/'metrics/run_statistics.json',stats)
    with (ROOT/'metrics/training_loss.csv').open('w',newline='') as f:
        fields=['step','loss','input_tokens','supervised_tokens','lr','grad_norm','seconds','elapsed_seconds'];w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in train)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    axs[0].plot([r['step'] for r in train],[r['loss'] for r in train],alpha=.25,lw=.7,color='#19608a')
    axs[0].plot(range(64,len(train)+1),np.convolve([r['loss'] for r in train],np.ones(64)/64,mode='valid'),color='#153e59',lw=2)
    axs[0].set(xlabel='New completed optimizer updates',ylabel='Mean per-example assistant/EOS loss',title='External mixed SFT: training loss')
    dev=[]
    for p in (ROOT/'metrics').glob('step_*_dev_summary.json'):
        s=json.loads(p.read_text());dev.append((int(s['arm'][5:]),s))
    dev.sort()
    for task in ['openbookqa','piqa','arc_challenge']:axs[1].plot([x[0] for x in dev],[x[1]['by_task'][task]['label_accuracy'] for x in dev],marker='o',label=task)
    axs[1].set(xlabel='New completed optimizer updates',ylabel='Dev label accuracy',ylim=(0,1),title='Dev only; held test scored after selection');axs[1].legend(fontsize=8)
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(ROOT/'metrics/learning_curves.png',dpi=160);plt.close(fig)
    lines=[]
    for task,c in results['paired_comparisons'].items():
        q=c['label'];g=c['generation'];ci=q['paired_bootstrap95']
        lines.append(f"| {task} | {q['before_correct']}/{q['n']} → {q['after_correct']}/{q['n']} | {g['before_correct']}/{g['n']} → {g['after_correct']}/{g['n']} | {100*q['difference']:+.2f} п.п. [{100*ci[0]:+.2f}; {100*ci[1]:+.2f}] |")
    b=results['summaries']['parent_EXT256'];a=results['summaries']['selected_EXT2'];ci=results['stratified_paired_bootstrap95'];status='подтверждён' if results['registered_improvement_criterion_met'] else 'не подтверждён'
    prior_notes='; '.join(f"{task}: постоянная train-majority метка {chr(65+p['constant_prediction_from_train'])} даёт {p['constant_correct']}/{p['n']}; самая частая метка модели занимает {100*p['selected_most_common_prediction_fraction']:.1f}% ответов" for task,p in results['training_label_prior_diagnostic'].items())
    gens=[json.loads(s) for s in (ROOT/'metrics/test_alpaca_generations.jsonl').read_text().splitlines()]
    examples=[]
    for old,new in zip(gens[:8],gens[8:]):
        assert old['id']==new['id'];examples.append(f"- `{old['id']}` — до: {json.dumps(old['generation'],ensure_ascii=False)}; после: {json.dumps(new['generation'],ensure_ascii=False)}")
    math=json.loads((ROOT/'metrics/math_results.json').read_text());assert math['completed'];mb=math['summary']['parent_EXT256'];ma=math['summary']['selected_EXT2']
    text=f'''# FlyGraph EXT2: внешнее дообучение и независимая оценка

**Цель превосходства над Qwen3.5-0.8B и MiniCPM-V-4.7-1B не установлена.** Обе точные reference-модели не измерены: официальный Hugging Face CONNECT заблокирован прокси (403). Числа из карточек и результаты других запусков не подставлялись. Для завершения сравнения нужен разрешённый доступ к официальным файлам либо полные локальные snapshots с весами/config/tokenizer/processor.

Выполнено **{len(train)} новых optimizer updates**, {stats['unique_training_records']:,} разных опубликованных примеров, {stats['input_tokens']:,} входных и {stats['supervised_tokens']:,} supervised токенов. Это продолжение реальных **100 028 328 параметров** EXT step256, warm start с новым AdamW. Старые 256 шагов не включены в число новых. Статус достижения фиксированных4096 шагов: **{results['training']['completed_fixed_step_count']}**; причина остановки `{results['training']['stop_reason']}`. Время обучения с dev/checkpoint: {results['training']['wall_seconds']/60:.2f} мин. GPU отсутствует, квота4CPU,16GiB RAM.

## Результат

По заранее зарегистрированному критерию рост качества **{status}**. Средняя парная разница label-ranking accuracy: **{100*results['mean_label_accuracy_difference']:+.2f} п.п.**,95% ДИ **[{100*ci[0]:+.2f}; {100*ci[1]:+.2f}]**. Alpaca held-out NLL: **{b['alpaca_nll']:.4f} → {a['alpaca_nll']:.4f}** (одинаковый токенизатор). Это сравнение с нашей предыдущей моделью, не с отсутствующими baseline.

| Внешний набор | Выбор по log p метки, до → после | Свободная генерация, до → после | Разница label accuracy и95% ДИ |
|---|---:|---:|---:|
{chr(10).join(lines)}

Основная метрика — log p одной допустимой буквенной метки при полном question/choices prompt. Отдельная свободная генерация — greedy по полному словарю, до16 новых токенов, остановка по EOS. Никакого solver, retrieval или ограничения словаря при генерации. Правильный формат сам по себе не означает правильное знание. Обе модели используют одинаковые входы/бюджет. Exact parser допускает отдельную заглавную букву с необязательной точкой/скобкой. Все768 сырых ответов каждого варианта сохранены.

Контроль постоянной метки, выбранной только по training labels: {prior_notes}. Это отдельная диагностическая проверка, а не подмена отсутствующих Qwen/MiniCPM. Преимущество над собственной слабой исходной версией без превосходства над таким контролем нельзя трактовать как убедительное улучшение рассуждения.

В {tie_audit['different_label_and_generation_predictions']} случаях label-ranking и свободная генерация выбрали разные буквы; во всех этих случаях сохранённые максимальные scores при BF16 inference точно равны: {tie_audit['all_disagreements_are_exact_top_score_ties']}. Ranking разрешает равенство по порядку вариантов A/B/C/D, greedy — по меньшему ID токена (D=780 раньше B=796). Правила и результаты после просмотра test не менялись; аудит использует только сохранённые ответы, без повторного inference.

Выбран checkpoint **нового шага {sel['best_step']}** только по минимуму среднего dev NLL на Alpaca/OpenBookQA/PIQA; ARC dev не входит в selection objective. Test открыт после фиксации выбора. В JSON есть Wilson-интервалы, точный McNemar, парный bootstrap10000 повторов, seed20261023. Один seed и ограниченный набор английских задач не доказывают общего превосходства; русский язык, код, мультимодальность и долговременная память этим запуском не оценены.

## Дополнительная независимая проверка математики

GSM8K,64 заранее зафиксированные задачи official test, seed20261024, без GSM8K train и без выбора checkpoint по этой метрике. Строгое совпадение конечного числа при одинаковом бюджете64 новых токена: **{mb['strict_correct']}/64 → {ma['strict_correct']}/64**. Корректный числовой формат: **{mb['strict_valid']}/64 → {ma['strict_valid']}/64**. Диагностическое совпадение последнего числа где-либо в тексте: **{mb['last_number_correct_diagnostic']}/64 → {ma['last_number_correct_diagnostic']}/64**; случайное число в рассуждении не объявляется правильным оформленным ответом. Достижение лимита без EOS: **{mb['budget_exhausted']} → {ma['budget_exhausted']}**. Ограниченный/неправильно оформленный вывод не доказывает отсутствие математического знания.

Это отдельная проверка переноса навыка, зарегистрированная до любых её model outputs, после начала основного обучения; она не добавлена задним числом в первичный трёхнаборный критерий. Источник и лицензияMIT: [GSM8K](https://github.com/openai/grade-school-math). Вопросы проверены на точное нормализованное вхождение в новый и прежний EXT train. Полные источники/seed/hash — `metrics/math_registration.json`, восстановление выборки — `src/prepare_math.py`, исходные ответы — `metrics/*_gsm8k_predictions.jsonl`.

## Диагностика исходного состояния

135/135 тензоров обоих исходных checkpoint загружаются строго и конечны; хеши совпали. На новых32 dev-записях L100EC имеет NLL13.4533, прежний EXT step256 —3.1245. Примеры до обучения: `22;F=22` и ` Shering sterating coners, cores`. У случайной модели инженерный dev8 NLL {diag['untrained_engineering_reference_dev8_nll']:.4f}; это диагностическая шкала, не обученный baseline.

Словарь8192 — частотные полные лексические фрагменты с UTF-8 fallback, **не BPE**. На новом dev **{100*diag['tokenizer']['byte_fallback_token_fraction']:.2f}% токенов побайтовые**; обратимость текста точная. Это снижает полезный объём текста на токен и вычислительную эффективность. Словарь сохранён ради совместимости с реальными весами; новый tokenizer или сброс embeddings не выдаётся за продолжение. Доступная история EC документирует203581 supervised tokens узкого curriculum; полный более ранний corpus не восстановлен, полноценное языковое pretraining не подтверждено. Эти наблюдения согласуются с узкой специализацией и слабым языковым основанием, но не исчерпывают причин низкого качества.

Исправлены два подтверждённых ограничения прошлого протокола: единая сериализация `User: ... / Assistant:` для train/dev/test и включение коротких ответов (8447 train-записей короче8 target tokens). Максимум512 train tokens, без обрезки содержимого. Отдельный аудит проверяет соответствие teacher-forced loss и cached label likelihood, а также causal cache/full-prefix logits на внешнем dev.

## Данные и разделение

- Train:8192 Stanford Alpaca,4096 OpenBookQA official train,4096 PIQA official train; все оригинальные записи сохранены. Никаких придуманных нами вопросов или gold-ответов. Seed20261022.
- Dev:128 новых Alpaca; по64 OpenBookQA/PIQA/ARC-Challenge. Alpaca dev/test исключают прежние1088 train/validation IDs.
- Test:256 OpenBookQA official test,256 ARC-Challenge official test,256 **зарезервированных записей official PIQA validation**,128 новых Alpaca. У PIQA это локальный независимый test из размеченного validation, не официальный скрытый test. ARC train не использовался. Прежний ARC-Easy128 не участвовал в выборе или обучении.
- Нормализованные совпадения новых train/dev/test отфильтрованы; выбранные held вопросы проверены в новых Alpaca train inputs/targets. В прежних1024 Alpaca training records найдены141 совпадение коротких generic stem-подстрок (17 разных слов длиной3–7 символов, например «pen»). Совпадений полного stem вместе со всеми вариантами ответа в одной прежней записи —0; это не доказывает отсутствие семантических пересечений. Новый substring-фильтр консервативен и мог исключать нерелевантные записи со случайным вхождением таких слов. Полный аудит: `metrics/prior_training_overlap_audit.json`. Один дубль вопроса в test заменён до оценки без изменения dev-хешей. Все768 test-вопросов различны. Не доказано отсутствие перефразов или неизвестной исторической pretraining contamination.
- Версии, оригинальные IDs, лицензии, точные SHA-256 и правила фильтрации: `data/manifest.json`, `DATA_LICENSES.md`. OpenCompass release0.2.2.rc1/20240207 служит транспортом оригинальных данных.

Источники: [Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [OpenBookQA](https://github.com/allenai/OpenBookQA), [PIQA и лицензия](https://github.com/ybisk/ybisk.github.io/tree/master/piqa), [ARC](https://huggingface.co/datasets/allenai/ai2_arc), [фиксированный архив](https://github.com/open-compass/opencompass/releases/tag/0.2.2.rc1). Alpaca и полученные research-веса имеют некоммерческие ограничения; лицензии наборов не смешиваются.

## Сохранность и проверка

Parent FP32 SHA-256: `{stats['parent_sha256']}`. Selected SHA-256: `{stats['selected_sha256']}`. Native model/AdamW — FP32, BF16 autocast, delta state FP32. LR1.5e-4, warmup64, cosine до10%, betas(0.9,0.95), WD0.01, clip1; четыре microbatch по одному, средний per-example assistant/EOS loss. Checkpoint каждые64 полных обновления, атомарное сохранение и точная проверка загрузки. Избыточные промежуточные recovery-файлы удалялись только после проверки преемника; сохранены первый64, лучший dev и последние checkpoints, остальные hash receipts.

Реальное воспроизведение новых шагов65–128 из native step64: все64 loss/grad norm совпали, все135 тензоров step128 **побитово одинаковы**. Это инженерная проверка, не дополнительное обучение финальной модели. Полный native checkpoint включает optimizer, RNG, cursor, schedule/config и идентичность данных. Политика публикации: обычные binary fragments менее90MB, SHA256 каждой части и сборки. Публикационный статус/commit/readback записываются отдельно; наличие этого отчёта само по себе не подтверждает push.

## Первые8 внешних test-примеров Alpaca

Порядок фиксирован заранее, примеры не выбраны по качеству. Бюджет32 новых токена одинаков для обеих моделей; это иллюстрации, не дополнительная ручная метрика.

{chr(10).join(examples)}
'''
    (ROOT/'REPORT_RU.md').write_text(text)
    print('Report and plots generated from completed raw metrics',flush=True)
if __name__=='__main__':main()
