# Отдельный dense/deep-thin контроль FlyGraph

Реализована и проверена **необученная архитектурная заготовка**, а не новая версия обученных EXT3-весов. Полноразмерные модели не создавались: параметры подсчитаны на meta-device и независимой формулой. Tiny-модель прошла проверки на CPU FP32. Optimizer-шагов0, pretraining/дообучение и оценка качества не проводились.

| Конфигурация | Параметры | Embedding | Блоки | Назначение |
|---|---:|---:|---:|---|
| dense_110m_existing_vocab | 110,922,048 | 4,718,592 | 30 | Основной контроль с прежним tokenizer8192 |
| dense_125m_vocab32k_template | 125,077,824 | 18,874,368 | 30 | Только шаблон; подходящего32k tokenizer здесь нет |
| tiny_smoke | 598,336 | 524,288 | 2 | Инженерные проверки, не языковая модель для использования |

В полноразмерных конфигах width576, FFN1536, GQA9 query/3 KV-heads, head_dim64, pre-RMSNorm eps1e-6, SwiGLU, RoPE theta10000, context2048. Input embedding и output projection используют один Parameter и одно хранилище. Все30 блоков независимы: sharing, MoE, delta и MLA отсутствуют. Инициализация локального FlyGraph std0.02 с residual-output scaling1/sqrt(2L); μP не реализован. QK-Norm не добавлен автоматически: это будущая отдельная проверка изменения функции.

Основа кода — собственные компоненты существующего FlyGraph, адаптированные в отдельном `src/model.py`. Научные ориентиры: [Qwen3§2](https://arxiv.org/html/2505.09388v1), [SmolLM2 config30×576/1536,GQA9/3](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json), [MobileLLM deep-thin](https://arxiv.org/abs/2402.14905v2), [MiniCPM training/scaling](https://arxiv.org/html/2404.06395v2). Внешний model code и веса не копировались. MobileLLM NC-реализация не использована. Полный локальный provenance и исходный SHA — `SOURCE_PROVENANCE.json`. Из перечня чужих идей не следует, что их сочетание лучше исходной модели.

**Несовместимость с EXT3 явная.** EXT3 имеет14 гибридных блоков width768; новый контроль30 dense-блоков width576 с другим пространством параметров. Для него требуется новая инициализация/pretraining или отдельная явно разработанная миграция/дистилляция. Нельзя загрузить EXT3 как strict resume, поменять только config или назвать smoke дообучением. Исходные EXT2/EXT3 артефакты сохранены без изменения.

**Проверено.** `src/smoke.py` проверяет shape и конечность logits, causal mask с реально отличающимся будущим, однотокенный и многотокенный cache/full эквивалент, сохранность входного cache, last_only, отсутствие cache по запросу, tied Parameter/storage, ошибки метаданных cache/context/targets, greedy cached-vs-full, ограничение длины генерации, конечные ненулевые градиенты каждого тензора, sparse-vs-dense masked loss, совпадение градиентов при activation checkpointing, native reload и сохранение tying. Всего46 проверок, из них18 — отдельные gradient comparisons. Точное содержимое/погрешности и выполненные команды сохраняются в metrics/logs.

Численные fixtures — первые два сохранённых исходных OpenBookQA dev-примера (`openbookqa:dev:7-1035`, `openbookqa:dev:1467`), лицензия Apache2. Точный release/member SHA и original records прилагаются. Инженерные операции с префиксами/будущим не являются новыми семантическими задачами или benchmark. Loss tiny-модели показывает только конечность вычислений; он не измеряет достигнутую языковую способность. Веса `artifacts/tiny_UNTRAINED_initial.safetensors` — лишь воспроизводимая random initialization seed2026102505. Backward не изменяет веса; optimizer отсутствует.

**Границы реализации.** Сейчас только обычные непаддированные causal sequences; произвольная padding/document mask и HF-compatible API не реализованы. Cache предназначен для inference; cached training явно отклоняется. CPU FP32 tiny-проверки не подтверждают BF16/GPU/длинный2048-контекст или обучение большого конфига. GQA временно повторяет KV перед SDPA; inference cache хранит только3 KV-heads. Attention prefill квадратичен по длине, обучение требует измерений. Дополнительная глубина увеличивает вычисления и KV: при2048 BF16 KV составит45MiB/sequence против14MiB у прежних семи attention-блоков. Это расчёт ёмкости cache, не измеренный peak RAM.

**Воспроизводимые команды из корня репозитория:**

```sh
python runs/dense_control_20261025/src/count_parameters.py --output /tmp/dense_counts.json
python runs/dense_control_20261025/src/smoke.py --readback --output /tmp/dense_smoke.json
python runs/dense_control_20261025/src/profile_step.py --config runs/dense_control_20261025/configs/tiny_smoke.json --sequence-length 32 --iterations 3 --output /tmp/dense_tiny_profile.json
```

Следующий фактический запуск перед выделением training budget — профиль большого **110M** конфига на имеющемся оборудовании. Эта команда подготовлена, но в текущем окне **не выполнялась**:

```sh
python runs/dense_control_20261025/src/profile_step.py --config runs/dense_control_20261025/configs/dense_110m_existing_vocab.json --sequence-length 64 --iterations 3 --output /tmp/dense_110m_cpu_profile.json
```

Профиль делает один warmup и три forward/backward без optimizer update. Он проверяет контракт tokenizer, сохраняет реальные времена/RSS и не обещает скорость3090 или другого не измеренного устройства. Запрос32k-конфига с8k tokenizer явно отвергается.

После измерений предложен **ещё не запущенный** pipeline-пилот на1,000,000 входных токенов, context256, microbatch1, accumulation4, wall limit2часа, native checkpoint каждые64 полных updates и в конце. Это проверка обучения/сохранения новой инициализации, не бюджет для достижения качества Qwen. Полный resume-safe training runner не входит в эту минимальную заготовку; фиктивная команда pretraining не приводится. До такого запуска нужны runner с optimizer/RNG/data cursor, опубликованный корпус с pinned license/revision, отдельный dev и новая независимая оценка. `NEXT_RUN_PLAN.json` фиксирует эти незавершённые предпосылки и бюджет.

Для честного вывода об архитектуре нужны новые dense и hybrid controls с сопоставимыми tokenizer/data/token-budget и измеренным compute; нельзя принять сравнение свежей random-модели и обученной EXT3 за доказательство пользы/вреда архитектуры. Изменены сразу глубина, ширина и тип блоков, поэтому одиночный такой контроль не изолирует эффект одного компонента.

Код, конфиги, raw check receipts, JSON/CSV и untrained tiny bytes публикуются отдельно в `runs/dense_control_20261025`. Publication status определяется `PUBLICATION.json`. Никакое значение качества и никакое превосходство над Qwen/MiniCPM не заявлены.
