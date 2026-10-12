# R04-H2: frozen 4B → bridge → frozen 0.8B (контроль передачи решения)

**Предрегистрация перед запуском H2.** Дата: 2026-10-12. Статус: `NOT_RUN`. Критерии исходного [R04 preregistration](../PREREGISTRATION.md) **не меняются**. Известные отрицательные результаты H1/H1b/H1c не могут быть исправлены этим новым экспериментом задним числом.

## Вопрос

Может ли замороженная Qwen3.5-0.8B **языково реализовать** содержание, выбранное замороженной Qwen3.5-4B, если видит только K плановых embedding-векторов, а не исходный вопрос, target текст или explicit verbosity tag? Будет ли fidelity к голове различима от решения задачи самой 0.8B?

## Фиксация источников

- Head: `Qwen/Qwen3.5-4B` SHA `c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97`; 32 слоя, hidden 2560, BF16, frozen.
- Writer: `Qwen/Qwen3.5-0.8B` SHA `23eda19ab0239bc12350efba411911b8eb89f486`; 24 слоя, hidden 1024, BF16, frozen.
- Только новый bridge 2560→K×1024 обучается. Primary `K=4`; абляции `K=1,8,16`.
- Содержание и степень подробности **выбирает голова**; writer не получает `question`, `style_id`, `length_id`, oracle digits или completion text на inference. Декодирование писателя остаётся autoregressive, что не является подтверждением одновременно вычисленного текста.

## Новые данные и защита от test leakage

До любого H2 обучения сгенерировать новый фиксированный набор (`seed=20261016`, candidate) 1200/200/300, по 4 заранее указанным RU/EN×arithmetic/code bucket. Исключить все ключи `(operator,a,b,c)` из 1700 H1, 128 H1b и 1700 H1c. Сохранить `dataset_sha256_canonical`, полный файл, SHA256 и неизменяемый split; зафиксировать внутри run-файлов до первого доступа к `test`.

Образцы из старых H1/H1b/H1c **test запрещено включать в train и validation** H2: их результаты уже известны. Особенно запрещено обучать на 50 teacher-correct задачах из первоначальных test/300. Вместо этого head на новом `train` сам генерирует решения; правильность помечается оракулом. Следует сохранить **все** ответы учителя, включая ошибки и невалидный формат. Не выбирать гиперпараметры по H2 test.

## Фазы проверки (все отделены в отчёте)

### H2-original: fidelity к исходной голове

- Извлечь frozen pre-answer h(x) от головы, без вставки правильного ответа или teacher-produced CoT в её prompt.
- Только на `train` получить внепоследовательные output targets от головы в зафиксированном generation mode. Нельзя «доклеивать» gold oracle в head states.
- Обучить мост `B(h)→soft K embeddings` по teacher-output token CE через замороженную 0.8B. Causal shift тестируется unit tests. Все trainable параметры — только B; у writer/head нулевые градиенты и параметры не меняются.
- Evaluate на **всех** 300 новых test + teacher-correct как вторичный условный срез. Две отдельные accuracy: (1) относительно сгенерированного ответа головы (faithful replication), (2) относительно истинного программного oracle (task correctness); отдельно invalid outputs.
- Сильные negative controls: zero/shuffled/contradictory plans; independent 0.8B question baseline; bridge с hash/structured features того же параметрического бюджета. Инерцию writer без плана фиксировать отдельно.

### H2a oracle-plan — инженерный положительный контроль, НЕ модель головы

- Вместо head hidden передать модулю soft-prefix **правильные два oracle-разряда** (супервизируемое представление 100 категорий), разрешая обучать только оракул→плановый bridge, writer frozen.
- `oracle plan` разрешён в этом режиме на train/val/test **только для проверки реализуемости интерфейса**. Его качество нельзя смешивать с H2-original: oracle не доступен настоящей head при работе пользователя.
- Если H2a не научится точно печатать две цифры, frozen writer+soft-prefix недостаточно даже при правильном плане — нужна иная схема conditioning, больше K или отдельный pretrained language-to-latent adapter.

### H2b explicit semantic bottleneck — отдельная абляция

- Добавить train-only supervised digit classifier на `h(x)` с target oracle; класс активирует 2 slot embeddings. Это **НОВЫЙ вычислительный решатель с супервизией**, а не безусловное доказательство готовности решения в `h`.
- Отчёт обязан сравнивать h→slot classifier с structured-operand baseline той же обучаемой мощности. Вся операция classifier входит во время работы головы/моста.

## Пороговые критерии — неизменяемые original R04/H2

- ≥75% правильных ответов на **полном** test.
- ≥80% правильных на teacher-correct subset (знаменатель, размер subset и доверительный интервал обязательны; subset **не используется в training**).
- ≥15 п.п. лучше сильного hash/structured контроля и ≥30 п.п. лучше shuffled-latent.
- Никаких итогов `success` до полного train/val/test, сохранённого exact checkpoint + AdamW/RNG и всех per-example outputs.
- Если голова на полной выборке решает около 16.7%, то **faithful reproduction** её ответов не может по определению дать 75% истинных ответов. Отдельно высокий task correctness студента тогда требует доказать, что он **решает дополнительно**, а не только «печатает».

## Ограничения VRAM и честная скорость

RTX 3090 24 GiB; сначала **раздельно** извлекать frozen h(x) на 4B, затем выгрузить голову и обучать bridge через frozen 0.8B, использовать BF16 с FP32 bridge, batch1–2, фиксировать peak reserved GiB. Это исследует transfer, но **не** H4 latency: для H4 нужно резидентное end-to-end выполнение head+bridge+writer без искусственного исключения загрузки/выгрузки.

## Последующие H3/H4

H3: Head принимает естественный запрос и выбирает краткую/подробную форму **сам**, не передавая writer style. H4: сквозное время и matched-quality. **Стартовать H3/H4 как отдельные измерения**, даже если H2 не подтверждён, но отрицательный quality gate H2 не позволяет считать скоростной выигрыш H4 доказанным.

## Status and publication

H2 код для GPU и результаты **не созданы** этим документом. Никакой claim об улучшении против Qwen3.5-4B нет. Ближайший артефакт — детерминированный generator нового набора и strict loader/baseline training source с CPU unit tests. Научная статья: [R02–R04 synthesis](../../SYNTHESIS_R02_R04_20261012_RU.md).
