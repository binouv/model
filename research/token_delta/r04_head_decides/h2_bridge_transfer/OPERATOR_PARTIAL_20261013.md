# R04-H2 — частичный GPU-прогон на RTX 3090 (13.10.2026)

**Статус: IN_PROGRESS_3_OF_12.** Это промежуточный отчёт по словам оператора, а не подтверждённый полный научный эксперимент. Полные исходные `run_receipts`, `metrics.json`, `predictions.jsonl`, бинарные checkpoints и feature-cache из NeoXider/model пока недоступны текущей GitHub-интеграции.

## Источник

- Оператор указал `NeoXider/model:r04-h2-partial` и `NeoXider/model:r04-h1-series` (force-update второй до `c102be9`).
- GitHub-интеграция этой беседы подключена как `binouv`; `NeoXider/model` и его refs отвечают 404. Попытка GitHub Actions анонимно прочитать публичную ветку ранее также не удалась. Это ограничение наблюдаемого доступа: байты **не импортированы**.
- Замороженные pretrained Qwen3.5-4B head и Qwen3.5-0.8B writer; только bridge обучаемый.
- Предрегистрация: 1700 задач, split 1200/200/300; canonical SHA `f06817743af405c27e6810bd859366d4d1b6c8c6714d02e699ff3f61f1ea65bf`.

## Предварительные результаты (НЕ вывод H2)

| Показатель | Сообщённое значение |
|---|---|
| Учитель: полная обработка 1700 случаев | Завершена |
| Teacher exact на H2 test | **49/300 = 16,33%** |
| Полных вариантов обучения | **3 из 12** |
| Завершённая группа | `head, K=4, seeds 11/22/33` |
| Writer: точность относительно истинного oracle | Приблизительно **4–5,3%**, точные per-seed знаменатели/числители ещё не проверены |
| Writer: fidelity/mimic к ответу головы | Приблизительно **13–15%**, точные определения по метрикам ещё не проверены |
| Writer: oracle accuracy на teacher-correct subset | По сообщению «почти ноль», точный числитель/знаменатель недоступен |
| Остальные варианты и контроли | **9/12 ещё не завершены** |

Не путать метрику correctness к oracle с `oracle-plan` — это отдельный *privileged* контроль, который ещё не подтверждён как выполненный. Оригинальные gates H2 (75% full-test, 80% teacher-correct, +15 п.п. над input-only и +30 п.п. над shuffled latent) **не меняются**. Отрицательный тренд первых трёх вариантов не означает формального завершённого `NOT_CONFIRMED`.

## Возобновление после перезагрузки

С теми же абсолютными/относительными расположениями данных, feature-cache и папки результатов, теми же `steps=400 batch=4 lr=0.0003`:

```powershell
& $py h2_bridge_transfer\src\h2_train.py --data h2_bridge_transfer\data --cache h2_bridge_transfer\cache --out h2_bridge_transfer\results\full --steps 400 --batch 4 --lr 0.0003
```

`h2_train.py` читает `run_receipts/<source>_K<k>_seed<seed>.json`, проверяет SHA256 соответствующего predictions JSONL, манифест и восстановленные байты настоящего checkpoint. Полностью завершённый вариант пропускается; незавершённый не учитывается. **Не очищать** `results/full/run_receipts/`, `checkpoints/`, `cache/` и не изменять `EXECUTION_LOCK.json`. Не делать force push в `binouv/model`.

## Что необходимо для финальной статьи

После 12/12: оригинальные `results/full/metrics.json` и `predictions.jsonl`, per-run receipts, checkpoints, optimizer/RNG, `cache/feature_manifest.json`, standalone writer baseline. Проверить честную точность на всех 300 примерах, отдельно teacher-correct, а также zero/shuffle/hash/oracle controls.

**Научный статус:** частичная исследовательская стадия, H2/H3/H4 не подтверждены, H4 matched-quality speedup не измерен.
