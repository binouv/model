# Как синхронизировать исходные H1/H1b/H1c артефакты с GitHub без прав NeoXider на binouv/model

**Дата:** 12 октября 2026. Это инструкция передачи существующих локальных Git-объектов, НЕ повторного запуска GPU. Сообщённые локальные коммиты: H1 `c8e3641`, H1b `33f20ab`, H1c `1e4696c`. При текущей проверке GitHub API ни один из этих коротких SHA не найден среди доступных коммитов удалённого `binouv/model`. Подключённый GitHub в чате работает под `binouv`, но не имеет доступа к файловой системе `D:\AI`.

## Предпочтительный вариант: форк + отдельные ветки

Создать в GitHub fork [binouv/model](https://github.com/binouv/model) в аккаунте **NeoXider** (при наличии форка этот шаг не нужен). В локальном PowerShell, из Git-клона, где существуют **все три SHA**, проверить:

```powershell
cd D:\AI\model_r04
git cat-file -t c8e3641
git cat-file -t 33f20ab
git cat-file -t 1e4696c
git show --stat --oneline c8e3641
git show --stat --oneline 33f20ab
git show --stat --oneline 1e4696c
```

Если каждый вывод `git cat-file -t` равен `commit` и в `git show --stat` нет токенов/паролей/HF-cache/виртуального окружения, добавить fork как remote:

```powershell
git remote add neoxider https://github.com/NeoXider/model.git
# Если remote neoxider уже существует, пропустить add и проверить:
git remote -v

git push neoxider c8e3641:refs/heads/r04-h1-raw-20261012
git push neoxider 33f20ab:refs/heads/r04-h1b-raw-20261012
git push neoxider 1e4696c:refs/heads/r04-h1c-raw-20261012
```

Пуш в **свой** форк под NeoXider не требует прав на binouv/model. Не выполнять force push. Если один SHA отсутствует в данном клоне, выполнять команду из того клона/worktree, который действительно содержит нужный коммит, с соответствующим `git remote`. Если Github запретит бинарный файл >100 MB, сначала проверьте проектное ограничение **90 000 000 bytes per file**, разбейте оригинал на байтовые части и сохраните manifest. Не заменяйте веса Git LFS pointer или хэшами.

После успешного пуша можно дать ссылки:

- `https://github.com/NeoXider/model/tree/r04-h1-raw-20261012`
- `https://github.com/NeoXider/model/tree/r04-h1b-raw-20261012`
- `https://github.com/NeoXider/model/tree/r04-h1c-raw-20261012`

Оттуда метрики и исходники станут доступными для сверки, после чего оформляется PR в `binouv/model`. Если ветки имеют самостоятельную историю, может понадобиться перенос научных артефактов в отдельную ветку от актуального `main` без потери SHA-checked файлов. Не сливать неконтролируемые бинарные изменения вслепую.

## Альтернатива: минимальная публикация для научной статьи

Если публиковать все ~80MB checkpoints пока неудобно, сначала приложить в чат или передать через fork следующие **оригинальные**, а не заново составленные файлы:

1. H1: `h1_metrics.json`, `h1_test_predictions.jsonl`, `head_test_receipt.json`, `head_test_predictions.jsonl`, `writer_test_receipt.json` и полный `features_manifest.json`.
2. H1b: `diagnosis_metrics.json`, `diagnosis_predictions.jsonl` и оригинальный аудит 300 H1 результатов.
3. H1c: `metrics.json`, `predictions.jsonl`, `feature_manifest.json`, набор checkpoint SHA256 и C2 calibration/step counts.

Из уже опубликованных GitHub-заготовок никакие отсутствующие raw outputs восстановить нельзя; короткий commit SHA не является самим файлом и не заменяет SHA256 оригинальных весов.

## Что будет проверено после синхронизации

- Сверить SHA256 каждого JSONL/manifest/компонента checkpoint.
- Заново посчитать all-test и per-stratum accuracy, invalid-format, thinking closure/truncation и все C0/C1/C2 gates по каждому seed.
- Проверить split/hash и отсутствие использования test для обучения/гиперпараметров.
- Сверить операционные записи с объявленными оригинальными порогами, обновить [общую статью](../SYNTHESIS_R02_R04_20261012_RU.md) с точными denominator, table, SHA и статусами; отрицательные результаты не скрывать.

**Состояние сейчас:** raw-артефакты ожидают передачи, исследовательская статья опубликована как v0.1 и явно помечает числа R04 как reported by operator.
