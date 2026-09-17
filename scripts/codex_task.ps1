# Передать задачу разработчику (Codex) и дождаться его отчёта.
#
#   .\scripts\codex_task.ps1 -Task 002
#   .\scripts\codex_task.ps1 -Task 001 -Review      # вернуть на доработку
#
# Работает в рабочей копии ветки agents/dev (C:\python\rag_textbook-codex).
# Человек вливает agents/dev в develop/graph_rag после приёмки.
#
# Модель и доступ задаёт профиль ~/.codex/rag-dev.config.toml. Скрипт
# не доверяет профилю на слово: читает шапку сессии и останавливается,
# если модель или уровень рассуждения не те.

param(
    [Parameter(Mandatory = $true)][string]$Task,
    [switch]$Review,
    # Продолжить оборванную сессию (например, по лимиту подписки) вместо новой:
    # Codex сохраняет контекст, и сделанное не повторяется. Id — в журнале.
    [string]$SessionId,
    [string]$Worktree = "C:\python\rag_textbook-codex",
    [string]$Branch = "agents/dev",
    [string]$Profile = "rag-dev",
    [string]$Model = "gpt-6-astra",
    [string]$Effort = "medium",
    # Экономия лимита. Каждый запрос к модели заново отправляет всю историю,
    # поэтому расход растёт как квадрат длины сессии: задача 006 сделала
    # 38 запросов с контекстом от 15 до 183 тыс. токенов — 4.25 млн на входе
    # и 70% пятичасового лимита. Сжатие истории и обрезка вывода команд
    # держат контекст маленьким.
    [int]$CompactAt = 60000,
    # ~12–15 тыс. символов кириллицы: хватает на раздел статьи, но не на
    # страницу целиком. Правило «до 8000 символов» в AGENTS.md — с запасом.
    [int]$ToolOutputTokens = 4000
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$taskFile = Join-Path $Worktree "tasks\$Task.md"
if (-not (Test-Path $taskFile)) { throw "Нет задачи: $taskFile" }

$current = (git -C $Worktree branch --show-current).Trim()
if ($current -ne $Branch) { throw "Рабочая копия на ветке '$current', ожидалась '$Branch'" }

$tempRoot = Join-Path $env:LOCALAPPDATA "Temp\codex-rag"
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $tempRoot "task-$Task-$stamp.log"
$final = Join-Path $tempRoot "task-$Task-$stamp.final.txt"

if ($SessionId) {
    $prompt = "Сессия была прервана (лимит или сбой). Продолжи задачу tasks/$Task.md с места остановки по правилам AGENTS.md, не повторяя сделанное. Сразу запиши в отчёт всё, что уже собрано, и дальше дописывай его по ходу."
} elseif ($Review) {
    $prompt = "Задача tasks/$Task.md возвращена на доработку. Исправь замечания из tasks/$Task-review.md по правилам AGENTS.md и дополни отчёт новым кругом."
} else {
    $prompt = "Выполни задачу tasks/$Task.md по правилам AGENTS.md."
}

Write-Host "==> Codex: задача $Task$(if ($Review) { ' (доработка)' })" -ForegroundColor Cyan
# Журнал пишется построчно и открыт для чтения: за Codex можно следить вживую
#   Get-Content -Wait -Encoding utf8 <журнал>
# (Set-Content держит файл запертым до конца и так не позволяет).
$stream = New-Object IO.FileStream($log, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::ReadWrite)
$writer = New-Object IO.StreamWriter($stream, (New-Object Text.UTF8Encoding $false))
$writer.AutoFlush = $true
Write-Host "    следить: Get-Content -Wait -Encoding utf8 '$log'" -ForegroundColor DarkGray
$started = Get-Date
# Модель и уровень передаются явно поверх профиля: профиль задаёт доступ,
# а выбор модели не должен зависеть от того, что лежит в общем config.toml.
# PowerShell 5.1 превращает любую строку stderr в ошибку, а Codex пишет туда
# служебные сообщения — на время вызова ошибки не прерывают скрипт.
$ErrorActionPreference = "Continue"
$effortArg = "model_reasoning_effort=`"$Effort`""
$budgetArgs = @("-c", "model_auto_compact_token_limit=$CompactAt", "-c", "tool_output_token_limit=$ToolOutputTokens")
if ($SessionId) {
    # У "exec resume" нет -p и -C: профиль передаётся общим флагом,
    # рабочая папка — текущим каталогом.
    Push-Location $Worktree
    codex -p $Profile exec resume -m $Model -c $effortArg @budgetArgs -o $final $SessionId $prompt 2>&1 |
        ForEach-Object { $writer.WriteLine("$_") }
    $resumeCode = $LASTEXITCODE
    Pop-Location
    $global:LASTEXITCODE = $resumeCode
} else {
    codex exec -p $Profile -m $Model -c $effortArg @budgetArgs `
        -C $Worktree -o $final $prompt 2>&1 |
        ForEach-Object { $writer.WriteLine("$_") }
}
$writer.Close()
$code = $LASTEXITCODE
$ErrorActionPreference = "Stop"
$seconds = [int]((Get-Date) - $started).TotalSeconds

# Шапка сессии: "model: ...", "reasoning effort: ...", "sandbox: ...".
$header = Get-Content $log -Encoding utf8 -TotalCount 40
$usedModel = ($header | Select-String '^model:\s*(.+)$' | Select-Object -First 1).Matches.Groups[1].Value
$usedEffort = ($header | Select-String '^reasoning effort:\s*(.+)$' | Select-Object -First 1).Matches.Groups[1].Value
$usedSandbox = ($header | Select-String '^sandbox:\s*(.+)$' | Select-Object -First 1).Matches.Groups[1].Value

Write-Host "    модель: $usedModel, рассуждение: $usedEffort, песочница: $usedSandbox"

# Расход по записи сессии: запросы, токены и остаток лимита подписки.
$sid = ($header | Select-String '^session id:\s*(\S+)' | Select-Object -First 1).Matches.Groups[1].Value
if ($sid) {
    $rollout = Get-ChildItem (Join-Path $env:USERPROFILE ".codex\sessions") -Recurse -Filter "*$sid.jsonl" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($rollout) {
        $requests = 0; $inTok = 0; $outTok = 0; $limits = $null
        foreach ($line in [IO.File]::ReadLines($rollout.FullName)) {
            if ($line.Contains('"token_usage_record"')) {
                $u = ($line | ConvertFrom-Json).payload.usage
                $requests++; $inTok += $u.input_tokens; $outTok += $u.output_tokens
            } elseif ($line.Contains('"rate_limits":{')) {
                $limits = ($line | ConvertFrom-Json).payload.rate_limits
            }
        }
        Write-Host ("    расход: запросов {0}, вход {1:N0} ток., выход {2:N0} ток." -f $requests, $inTok, $outTok)
        if ($limits -and $limits.primary) {
            Write-Host ("    лимит: 5 ч — {0}%, неделя — {1}% (тариф {2})" -f $limits.primary.used_percent, $limits.secondary.used_percent, $limits.plan_type)
        }
    }
}
Write-Host "    код: $code, секунд: $seconds, журнал: $log"

$bad = @()
if ($usedModel -ne $Model) { $bad += "модель '$usedModel' вместо '$Model'" }
if ($usedEffort -ne $Effort) { $bad += "рассуждение '$usedEffort' вместо '$Effort'" }
if ($code -ne 0) { $bad += "codex завершился с кодом $code" }

Write-Host "`n--- итог Codex:" -ForegroundColor Cyan
if (Test-Path $final) { Get-Content $final -Encoding utf8 } else { Write-Host "<нет итогового сообщения>" }

if ($bad) {
    Write-Host "`nОСТАНОВ: $($bad -join '; ')" -ForegroundColor Red
    exit 1
}
Write-Host "`nОтчёт: tasks\$Task-report.md. Приёмка — собственной проверкой, не по отчёту." -ForegroundColor Green
