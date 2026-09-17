# Очередь задач Codex: строго по одной, с ожиданием сброса лимита.
#
#   .\scripts\codex_queue.ps1 -Items "006","007","005@<session-id>" -StartAt "21:02"
#
# Элемент "NNN" — новая задача, "NNN@<id>" — продолжение оборванной сессии.
# Параллельный запуск трёх задач 2026-09-16 исчерпал лимит подписки за
# четыре минуты, поэтому только последовательно.
#
# Ход пишется в %LOCALAPPDATA%\Temp\codex-rag\queue-status.log построчно:
#   START / DONE / FAIL / LIMIT / QUEUE_END. На LIMIT очередь останавливается —
# остаток продолжают после следующего сброса (id сессии есть в строке).

param(
    [Parameter(Mandatory = $true)][string[]]$Items,
    [string]$StartAt
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Join-Path $env:LOCALAPPDATA "Temp\codex-rag"
$status = Join-Path $root "queue-status.log"
$runner = Join-Path $PSScriptRoot "codex_task.ps1"

function Say([string]$line) {
    $stamped = "$(Get-Date -Format 'HH:mm:ss') $line"
    Add-Content -Path $status -Value $stamped -Encoding utf8
}

if ($StartAt) {
    $target = [datetime]::ParseExact($StartAt, "HH:mm", $null)
    if ($target -lt (Get-Date)) { $target = $target.AddDays(1) }
    Say "WAIT до $($target.ToString('HH:mm'))"
    while ((Get-Date) -lt $target) { Start-Sleep -Seconds 30 }
}

# Через -File список приходит одной строкой "006,007" — разбираем сами.
$Items = @($Items | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
foreach ($item in $Items) {
    $task, $session = $item -split "@", 2
    Say "START $task$(if ($session) { " (продолжение $session)" })"
    $argList = @("-ExecutionPolicy", "Bypass", "-File", $runner, "-Task", $task)
    if ($session) { $argList += @("-SessionId", $session) }
    $out = & powershell @argList 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE

    $logLine = $out | Where-Object { $_ -match "журнал: (.+\.log)" } | Select-Object -First 1
    $log = if ($logLine -match "журнал: (.+\.log)") { $Matches[1] } else { $null }
    $sid = if ($log) { (Select-String -Path $log -Pattern '^session id:\s*(\S+)' | Select-Object -First 1).Matches.Groups[1].Value }
    $limited = $log -and (Select-String -Path $log -Pattern "usage limit" -Quiet)

    if ($limited) {
        Say "LIMIT $task session=$sid — очередь остановлена"
        break
    } elseif ($code -eq 0) {
        Say "DONE $task session=$sid"
    } else {
        Say "FAIL $task code=$code session=$sid журнал=$log"
    }
}
Say "QUEUE_END"
