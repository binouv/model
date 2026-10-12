# R04-H1 — локальный запуск на RTX 3090 (Windows / PowerShell)

**Выбор: начинаем H1 сейчас.** Не блокируем исследование из-за 403 локального `NeoXider`. Для H1 не изменяем веса Qwen3.5-4B/0.8B, не обучаем мост; только читаем скрытые состояния и обучаем маленькие диагностические классификаторы.

## 0. Сохранить свой локальный smoke-коммит и получить H1-код

В GitHub коннектор этого чата подключён как `binouv` и может публиковать исходники напрямую в `binouv/model`. На вашем Windows Git credentials записаны как `NeoXider`, поэтому прямой push от него в чужой репозиторий закономерно даёт 403. **Не удаляйте локальный коммит `72922ba` и не используйте `reset --hard`.**

```powershell
cd D:\AI\model_r04
# Сделать ссылку на имеющийся коммит, даже если HEAD позже переключится:
git branch r04-smoke-local 72922ba
# Скачивание публичного main не требует прав записи:
git fetch origin main
# Отдельная чистая рабочая ветка от актуального опубликованного H1:
git switch -c r04-h1-local origin/main
cd research\token_delta\r04_head_decides
```

Чтобы опубликовать **точный** smoke.json позже: переключить `gh auth` на аккаунт с правами записи (`gh auth status`, затем `gh auth switch --user binouv` при наличии этого аккаунта), создать ветку от `origin/main`, `git cherry-pick 72922ba`, проверить `git show --stat HEAD` и запушить ветку, затем PR. Если у вас только `NeoXider`, сделать fork и PR из fork. Не копируйте summary из чата вместо исходного `results/smoke.json` при публикации численных результатов.

## 1. Подготовить данные — запускать первым, без GPU

Вы уже подтвердили `torch==2.10.0+cu128` и `transformers==5.13.0`; используйте **свою текущую `.venv`** вместо нового окружения.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m unittest discover -s h1\tests -v
& $py h1\src\h1_dataset.py --out h1\data
Get-Content h1\data\dataset_manifest.json
```

Ожидается `1700` задач, `train=1200, val=200, test=300`, seed `20261012`. Каждый ответ вычисляется программным оракулом; тесты проверяют исполнение сгенерированного Python-кода.

## 2. Небольшой GPU-пилот (инженерный, не H1)

```powershell
$env:HF_HOME = 'D:\AI\hf_cache'
& $py h1\src\h1_pilot.py --data h1\data --samples 8 --output h1\results\h1_pilot_gpu.json
```

Проверяются `hidden_states[16]`, `[24]`, `[32]` размерностью `[8,2560]`, BF16/CUDA, строгая генерация на 8 **обучающих** примерах и пик VRAM ниже 22 GiB. Пилот **не** используется для утверждения об успешности H1 и не позволяет менять пороги.

## 3. Полное извлечение скрытых состояний и прямые baseline

```powershell
& $py h1\src\h1_extract.py `
  --data h1\data --out h1\cache --batch 4 --shard 50 `
  --run-writer-baseline --max-peak-gib 22
```

- Замороженная **Qwen3.5-4B** один раз читает каждый из 1700 вопросов. Сохраняются BF16-векторы слоёв **16/24/32**, до появления любых токенов ответа.
- Кеш пишется shards по 50 примеров; при прерывании можно безопасно повторить ту же команду: готовые shards верифицируются по метаданным и SHA256, остальные вычисляются.
- Прямой greedy **4B** и отдельно **0.8B** запускаются только на 300 тестовых вопросах, без использования латента. Каждый ответ, включая ошибки/форматирование, записывается для анализа.
- Готовые результаты: `h1/cache/h1_features.safetensors`, `features_manifest.json`, `head_test_predictions.jsonl`, `head_test_receipt.json`, `writer_test_*.json*`.

**Не считайте весь этап завершённым**, пока оба baseline receipt не сохранены и не проверены.

## 4. Диагностические пробы (считают только ответ из скрытого состояния)

```powershell
& $py h1\src\h1_probes.py `
  --data h1\data --cache h1\cache --out h1\results\full `
  --steps 600 --batch 96 --device cuda --seeds 11 22 33
```

Все кандидаты обучаются **только** на 1200 train; `200 val` выбирает архитектуру/слой, `300 test` проверяет уже выбранную. Анализируются раздельно арифметика, Python code-output и RU/EN. Все веса и состояния AdamW/RNG сохраняются, даже если результат отрицательный.

**Файл ответа**: `h1/results/full/h1_metrics.json` с `h1_conclusion`. Возможные статусы: `CONFIRMED_under_registered_gates`, `NOT_CONFIRMED_under_registered_gates`, `INCOMPLETE_STANDALONE_BASELINES`. Читайте `gate_per_seed`, отдельные `digit0/digit1`, `teacher_baseline`, `writer_only_baseline`. Не интерпретируйте единичный токен как длинный готовый ответ.

## 5. Сохранение и Git-публикация

- Исходники/PREREGISTRATION фиксируются **до GPU H1**. После завершения сохраняйте все реальные артефакты и SHA256; никогда не пушьте мнимые результаты.
- При локальном 403 от `NeoXider`: `gh auth status`, `gh auth switch --user binouv` (при реальном доступе), `git remote -v`; иначе push в собственный fork + PR. Можно приложить exact `results/smoke.json` сюда и использовать уже подключённый GitHub `binouv` для публикации.
- Не коммитьте HF cache, полные веса Qwen или venv; публикуются только выводы эксперимента и сохранённые обученные маленькие probe checkpoints. Ограничение файла: **90 000 000 байт**. Большие файлы разбить байт-в-байт через уже существующий checkpoint_parts.py из репозитория.
- Незаконченный run, GPU OOM и `KeyboardInterrupt` сохраняются как `incomplete`, но не считаются положительным/отрицательным законченным тестом.

### Что отправить для анализа

`h1/results/h1_pilot_gpu.json`, затем `h1/cache/{features_manifest.json,head_test_receipt.json,writer_test_receipt.json}`, наконец `h1/results/full/{h1_metrics.json,h1_test_predictions.jsonl}` и SHA checkpoint-манифеста.