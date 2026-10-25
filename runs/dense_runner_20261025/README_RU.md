# Проверенный минимальный resumable runner для dense-контроля

Открытый пункт закрыт: `src/train.py` выполняет CPU FP32 next-token LM training и точно восстанавливает native checkpoint. Проверка проведена только на tiny-модели598,336 параметров. Это инженерный smoke, не pretrained/полезная языковая модель и не продолжение EXT3. Полный110M pretraining не запускался.

**Доказательство.** Непрерывные6 шагов сравнивались с отдельными процессами3 шага→checkpoint→resume ещё3 шага. Итог355 входных токенов, пересечена граница эпохи. Побитово совпали все18 model tensors и весь training state: AdamW, RNG Python/NumPy/Torch, sampler order/index/offset/epoch/shuffle-RNG, global step/token count и все6 losses/gradient norms. Losses:9.0041304,8.9636164,8.8833828,8.9580231,8.8266201,8.8592815 — только инженерный журнал, не оценка качества. Прогон из восстановленных бинарных частей дал тот же результат. Отдельный бюджет70 остановился ровно на70 входных токенах после2 updates. Изменение source data и повреждение model bytes вызвали отказ до resume.

**Сохранение.** Checkpoint содержит native FP32 safetensors, AdamW/state в PyTorch-файле, точные model/train configs, SHA данных/tokenizer/исполняемых исходников, RNG, sampler, scheduler contract, scaler=None и CUDA RNG=None (CPU без AMP). Каждый update завершён целиком; accumulation1. Файлы fsync, проверяются strict reload и SHA, затем публикуется атомарно новая checkpoint-папка. Предыдущие checkpoints не перезаписываются. Native bytes дополнительно разбиты без изменения точности на части максимум80,000,000B; manifest сохраняет порядок, размеры, SHA, dtype/count, число параметров и завершённый шаг. Части не самостоятельные safetensors: сначала восстановление.

Resume отказывается при изменении model/train config, dataset/tokenizer/source SHA или версии Torch. Сначала проверяются SHA native-файлов, затем модель strict load, AdamW, sampler и RNG. Сохранён constant-LR schedule; другие scheduler не подразумеваются. Лимиты token/steps глобальны при resume; wall limit действует на одну команду запуска. Ранний wall/controlled stop получает interrupted/recovery статус, а не завершённый benchmark. Неожиданная ошибка оставляет последний проверенный checkpoint; незавершённый update не считается выполненным.

**Данные.** JSONL содержит `text` либо исходные `prompt`+`completion`; обучение следующему токену идёт по всему документу с BOS/EOS. Это LM objective, не assistant-only SFT. Окна длиной sequence_length сбрасывают attention state; sampler сохраняет позицию внутри документа и перемешивание между эпохами. Здесь используются только два исходных опубликованных OpenBookQA dev-примера как численные fixtures, Apache2; версии/SHA/ID в data/manifest.json. Они не являются независимым quality test и не используются для заявления роста качества. Внешний model code/NC-веса не импортированы: модель точно скопирована из уже проверенного собственного dense_control scaffold.

**Воспроизвести tiny proof из корня репозитория:**

```sh
python runs/dense_runner_20261025/src/verify_resume.py --workspace /tmp/dense_runner_replay_new --output /tmp/dense_runner_replay.json
```

Для проверки уже скачанных native bytes:

```sh
python runs/dense_runner_20261025/src/verify_resume.py --saved-root runs/dense_runner_20261025/artifacts/local_smoke --workspace /tmp/dense_runner_readback_new --output /tmp/dense_runner_readback.json
```

**Запустить tiny runner самостоятельно** (output должен быть новым):

```sh
python runs/dense_runner_20261025/src/train.py --model-config runs/dense_runner_20261025/configs/tiny_smoke.json --train-config runs/dense_runner_20261025/configs/tiny_train.json --data runs/dense_runner_20261025/data/external_dev_fixtures.jsonl --tokenizer runs/dense_runner_20261025/data/tokenizer.json --output /tmp/dense_tiny_first --stop-after 3
python runs/dense_runner_20261025/src/train.py --model-config runs/dense_runner_20261025/configs/tiny_smoke.json --train-config runs/dense_runner_20261025/configs/tiny_train.json --data runs/dense_runner_20261025/data/external_dev_fixtures.jsonl --tokenizer runs/dense_runner_20261025/data/tokenizer.json --output /tmp/dense_tiny_resumed --resume /tmp/dense_tiny_first/step_000003
```

**Следующий полноразмерный pipeline-пилот пока НЕ выполнялся.** Конфиг110,922,048 параметров; token budget1,000,000, context256, batch1/accumulation1, AdamW lr3e-4 constant, checkpoint64, wall limit7200s/команду. Точная поддерживаемая команда ниже использует уже сохранённый внешний train JSONL; перед ней нужны измерение throughput и свободное место:

```sh
python runs/dense_runner_20261025/src/train.py --model-config runs/dense_runner_20261025/configs/dense_110m_existing_vocab.json --train-config runs/dense_runner_20261025/configs/dense_110m_pilot_UNRUN.json --data runs/ext2_20261022/data/train.jsonl --tokenizer runs/dense_runner_20261025/data/tokenizer.json --output /workspace/dense_110m_future_pilot
```

Это новая инициализация, не FT существующей FlyGraph. У источников train свои лицензии, включая некоммерческую Alpaca в EXT2; они не снимаются этой командой. Для quality-эксперимента отдельно фиксируются dev/test и сравнение с контролем; уже раскрытые тесты не используются для выбора.

**Материальные ограничения.** CPU FP32, один процесс, microbatch1, без AMP/distributed/gradient accumulation>1 и без встроенной dev-selection/evaluation. Только constant LR; WSD/μP не реализованы. Полноразмерная скорость и GPU/3090 не измерены. Runner сохраняет все recovery checkpoints, автоматического pruning нет: один110M checkpoint с native model+AdamW и дублирующими parts порядка2.7GB, поэтому текущего свободного места среды недостаточно для указанного полного пилота. Сначала требуется подходящее локальное хранилище/явная проверенная retention-политика. Никакие ресурсы не арендовались. Минимальный runner проверен по перечисленным функциям; это не production pretraining platform.

Текущая публикация содержит реальные tiny model/state bytes всех выполненных smoke-веток и их manifests, CSV/JSON, логи, код, конфиги и source/data identities. Pause наstep3 является recovery-артефактом завершённой инженерной проверки6steps, не отдельным завершённым training run. `PUBLICATION.json` подтверждает clean remote readback и replay из скачанных частей. Обученные EXT3-веса и прежние отчёты не менялись.
