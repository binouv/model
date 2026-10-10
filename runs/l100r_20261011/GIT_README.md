# FlyGraph L100R: завершённое парное продолжение 100M

**Состояние опыта: completed_paired_experiment. Общая цель превзойти Qwen3.5-0.8B: НЕ достигнута.**
Это продолжение собственной FlyGraph, не обучение Qwen и не новая модель с нуля.

## Фактически выполнено

Восстановлены реальные BF16-веса L100M step1024: 100028328 параметров.
SHA256 весов предка: `b2112e21dcc653780da5c6f36137c0deab3f2fcc01bb04662eb518cf4a07b3ec`.
Хеши архивов, tokenizer и 10 файлов исходных данных проверены. Полный optimizer
предка не найден: использован BF16-to-FP32 warm-start с новым Adafactor.

Завершены две ветки по384 optimizer updates, всего768 новых научных обновлений:
`trace` — проверенная промежуточная трасса и ответ; `final` — только ответ.
Batch4, одинаковые1536 предъявлений задач и порядок batch IDs, один предок/seed.
Trace:107448 input и31533 supervised tokens; final:84704 input и8789 supervised.
Это НЕ сравнение при равных FLOPs: trace имеет больше target tokens.
Внутри trace выполнен реальный выход процесса на192 и resume новым процессом с193.
Два отброшенных replay-updates193–194 воспроизвели loss, grad norm, lr и batch IDs;
они, как и два preflight-updates, не включены в768 scientific updates.

Исполнение локальное, CPU, torch2.10.0+cpu; CUDA отсутствует. Все135 тензоров
обеих веток изменились и конечны. 12 инженерных тестов прошли. Реализованы
device-aware trainer, проверяемый optimizer/RNG resume, chunked128 delta без
обрыва градиента между chunks, защита от повторного обучения завершённой ветки.
CUDA forward/backward и скорость RTX3090 не проверены. Chunked delta — собственная
реализация, не FLA/Triton. Activation checkpointing проверен на малой конфигурации,
но в основном100M-run выключен.

## Независимо пересчитанные результаты

Те же500 синтетических held-out задач предка; это не новые внешние benchmarks.
Модель получает только prompt, без solver, gold или списка ответов. Greedy по
всем8192 токенам, максимум48 новых токенов, EOS обязателен, фиксированный strict
парсер F=<integer> или T=<trace>;F=<integer>. Последнее произвольное число не ищется.

| Модель | IID /200 | Surface /100 | Extrapolation /100 | CF задачи /100 | Полные CF пары /50 | Всего /500 |
|---|---:|---:|---:|---:|---:|---:|
| Предок L100M |1|0|0|0|0|1 (0.2%)|
| L100R trace |4|1|0|0|0|5 (1.0%)|
| L100R final |5|1|0|1|0|7 (1.4%)|

Все1500 основных ответов завершились EOS. Формат корректен в499/495/500 случаях
соответственно, но формат не равен правильности. Преимущество trace над final
не подтверждено: IID−0.5п.п., полные CF пары0п.п.; verdict
`NOT_CONFIRMED_IN_REGISTERED_PROTOCOL`. Число7/500 не означает практически
пригодную модель, общий рост интеллекта или превосходство над Qwen.

Все1500 ответов и768 строк updates отдельно пересчитаны stdlib-аудитором.
Первоначальный reload-тест менял состав батчей и обнаружил одну разницу trace
токенов при одинаковом финальном ответе. Проверка исправлена на оригинальные
батчи:69 записей для каждой модели совпали по token IDs/EOS. Primary predictions,
метрика и обучение не переписаны. Старый evaluator и диагностическая запись сохранены.

## Qwen и незакрытые части

Новый Qwen0.8 baseline не выполнен: локального сервера и библиотек нет, загрузка
весов в среду не удалась. Исторический отменённый CI-run37950290677 восстановлен
отдельно как ZIP artifact11628510070: лишь9 запросов из20. Его нельзя объявлять
завершённым, смешивать с500-case таблицей или считать отсутствующие11 ошибки.
SHA256 исторического ZIP: `b0402b42b4956de8b64a9ee1cfb224cd6b13802719a2cd569f4c8d6369935908`.
Сами Qwen weight bytes этим ZIP не подтверждены.

Старые L100C/L100P controls не восстановлены и не объявлены закрытыми.
Не выполнены внешние Lighteval/EvalPlus evaluations, генерация Python с тестами,
общая RU/EN-генерация, CUDA-validation, новые memory/recurrent механизмы.
Этот run закончен, но исходная исследовательская цель остаётся открытой.

## Реальные checkpoints и ZIP-доставка

FP32 trace model.safetensors SHA256:
`5cc84ca9cdb4ff51c8cfbb0fa6b349645dc1b8f006aa0fa4736dbf84528e4113`.
FP32 final model.safetensors SHA256:
`51d024b5a63fc630c3acb636e293b3385f666e8eadcb5a210182bbf2d7e73f4f`.
Каждый файл400125984 байта; optimizer/RNG сохранены отдельно.

В ответе пользователю поставлены четыре проверенных ZIP:
- Source_Results:13252037байт, SHA256`3ec414867002a8ea835cf1d15c49d5b486b17c03d67aa5197d6b678806f79b74`.
- trace_FP32_Resume:373016522байт, SHA256`2dcd18ff57d5004cfa434e125afed3460a441033f9435dc196a604a4a12172c2`.
- final_FP32_Resume:372975036байт, SHA256`8d9a1ad227e0da27158d624a43876fe9817a8de7c2e18418de7f0bd52de0272d`.
- trace_BF16:160435956байт, SHA256`2ed01fa4f13b52947b3c61b6e92c8dde383ab7c921d8e64c9db23449ebf6012d`.

Полные имена файлов и хеши: `DOWNLOAD_MANIFEST.json`. ZIP прочитаны обратно,
каждый член проверен по SHA256. FP32 Resume содержит веса, optimizer/RNG, config,
tokenizer и model source. BF16 — экспорт для инференса, не exact-resume checkpoint;
основная оценка относится к FP32-весам с BF16 autocast.
**В этом Git-коммите нет больших весов; GitHub Release с ними не создан.**

## Получение исходников и всех raw records из Git

Git blobs в source_parts содержат точные27 исходных файлов; records_parts —
точные50 файлов config/metrics/reports/manifests в lossless-представлении.
Проверенное побайтовое восстановление: `GIT_REPLAY_RECEIPT.json`.

```bash
python bootstrap.py --source-only
# Сначала распаковать Source_Results ZIP сюда, включая data/source_data и tokenizer.
python bootstrap.py --records --data data/source_data
```

Source_Results ZIP уже содержит обычные распакованные файлы и полные данные.
Records decoder проверяет dictionary/data/tokenizer и SHA256 каждого восстановленного
файла. Он не обучает модель, не генерирует новые ответы и не подменяет наблюдённые
loss/grad_norm/timing. Восстанавливаются те же исходные байты JSON/JSONL.
Имеющиеся изменённые файлы не перезаписываются. Полный отчёт после раскрытия:
`reports/REPORT_RU.md`; primary numbers:`metrics/SUMMARY.json`; точки продолжения:
`reports/NEXT_STATE.json`. Не повторять384 законченных updates ни одной ветки.
