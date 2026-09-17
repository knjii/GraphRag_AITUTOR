# Отчёт по задаче 002

Изменён только `tasks/002-report.md`: записаны результаты проверки окружения. Код и тесты проекта не менялись, пакеты не устанавливались, коммиты не создавались.

## 1. Временная папка

Команда (PowerShell):

```powershell
Write-Output "TEMP=$env:TEMP"
```

Вывод:

```text
TEMP=C:\Users\Максим\AppData\Local\Temp\codex-rag
```

## 2. Тесты без --basetemp

Команда:

```powershell
C:\python\rag_textbook\.venv\Scripts\python.exe -m pytest tests/test_inspect_cell.py -q
```

Вывод (как получен):

```text
...............................                                          [100%]
31 passed in 0.40s
```

Код завершения: 0.

## 3. Запись, чтение и удаление в %TEMP%

Команды:

```powershell
$probePath = Join-Path $env:TEMP 'probe-002.txt'
if (Test-Path -LiteralPath $probePath) { throw 'Пробный файл уже существует' }
Set-Content -LiteralPath $probePath -Value 'Проверка окружения 002' -Encoding UTF8 -ErrorAction Stop
Get-Content -LiteralPath $probePath -Encoding UTF8
Remove-Item -LiteralPath $probePath -ErrorAction Stop
Write-Output "Exists after deletion: $(Test-Path -LiteralPath $probePath)"
```

Вывод:

```text
Проверка окружения 002
Exists after deletion: False
```

## 4. Запись в родительскую папку

Команды:

```powershell
$parentProbe = 'C:\Users\Максим\AppData\Local\Temp\probe-002.txt'
if (Test-Path -LiteralPath $parentProbe) { throw 'Пробный файл уже существует' }
try {
    Set-Content -LiteralPath $parentProbe -Value 'Проверка границы доступа 002' -Encoding UTF8 -ErrorAction Stop
    Write-Output 'Parent write: SUCCESS'
} catch {
    Write-Output "Parent write: DENIED/ERROR: $($_.Exception.Message)"
} finally {
    if (Test-Path -LiteralPath $parentProbe) {
        Remove-Item -LiteralPath $parentProbe -ErrorAction Stop
        Write-Output 'Parent probe removed'
    }
    Write-Output "Parent probe exists: $(Test-Path -LiteralPath $parentProbe)"
}
```

Вывод:

```text
Parent write: DENIED/ERROR: Access to the path 'C:\Users\Максим\AppData\Local\Temp\probe-002.txt' is denied.
Parent probe exists: False
```

Получен ожидаемый отказ в доступе. Файл не создан, удаление не потребовалось.

## 5. Сеть на чтение и версия pytest

Первый запуск:

```powershell
C:\python\rag_textbook\.venv\Scripts\python.exe -m pytest --version
$pytestMetadata = Invoke-RestMethod -Uri 'https://pypi.org/pypi/pytest/json' -Method Get
Write-Output "PyPI pytest: $($pytestMetadata.info.version)"
```

Вывод:

```text
pytest 9.1.1
Invoke-RestMethod : Базовое соединение закрыто: Непредвиденная ошибка при приеме.
At line:3 char:19
+ ... tMetadata = Invoke-RestMethod -Uri 'https://pypi.org/pypi/pytest/json ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (System.Net.HttpWebRequest:HttpWebRequest) [Invoke-RestMethod], WebExc
   eption
    + FullyQualifiedErrorId : WebCmdletWebResponseException,Microsoft.PowerShell.Commands.InvokeRestMethodCommand

PyPI pytest:
```

Повторное чтение того же URL через стандартную библиотеку Python:

```powershell
C:\python\rag_textbook\.venv\Scripts\python.exe -c "import json, urllib.request; print('PyPI pytest:', json.load(urllib.request.urlopen('https://pypi.org/pypi/pytest/json', timeout=30))['info']['version'])"
```

Вывод:

```text
PyPI pytest: 9.1.1
```

Установленная версия совпадает с последней версией из поля `info.version` PyPI: **9.1.1**. Чтение по сети работает через Python. Причина ошибки соединения в PowerShell не установлена.

Источник: [метаданные pytest на PyPI](https://pypi.org/pypi/pytest/json). Другие внешние источники не использовались.

## 6. Состояние рабочей копии

До проверок команда `git status --short` выдала:

```text
 M scripts/inspect_cell.py
?? .claude/
?? AGENTS.md
?? scripts/codex_task.ps1
?? tasks/
?? tests/test_inspect_cell.py
```

Для различения файлов внутри неотслеживаемых папок также выполнена команда `git status --short --untracked-files=all` до создания отчёта:

```text
 M scripts/inspect_cell.py
?? .claude/skills/codex-dev/SKILL.md
?? AGENTS.md
?? scripts/codex_task.ps1
?? tasks/001-report.md
?? tasks/001-review.md
?? tasks/001.md
?? tasks/002.md
?? tests/test_inspect_cell.py
```

После создания отчёта команда `git status --short` выдала:

```text
 M scripts/inspect_cell.py
?? .claude/
?? AGENTS.md
?? scripts/codex_task.ps1
?? tasks/
?? tests/test_inspect_cell.py
```

Итоговый вывод `git status --short --untracked-files=all`:

```text
 M scripts/inspect_cell.py
?? .claude/skills/codex-dev/SKILL.md
?? AGENTS.md
?? scripts/codex_task.ps1
?? tasks/001-report.md
?? tasks/001-review.md
?? tasks/001.md
?? tasks/002-report.md
?? tasks/002.md
?? tests/test_inspect_cell.py
```

Единственный новый файл относительно исходного состояния — `tasks/002-report.md`. Посторонних неотслеживаемых файлов после проверок нет. Оба пробных файла отсутствуют.

## Ограничения и невыполненное

Пункты 1–6 выполнены. Ошибка `Invoke-RestMethod` сохранена в отчёте; успешное чтение через Python подтверждает доступность PyPI, но не исправность сетевого стека PowerShell. Причина этой ошибки не исследовалась. Исходные изменения рабочей копии оставлены без изменений.
