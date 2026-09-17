# Смотреть вживую, что делает Codex, пока им управляет Claude.
#
#   .\scripts\codex_watch.ps1
#
# Ждёт появления журнала очередной задачи (codex_task.ps1 пишет их в
# %LOCALAPPDATA%\Temp\codex-rag) и печатает его по мере записи. Когда задача
# закончилась и началась следующая, переключается на новый журнал сам.
# Выход — Ctrl+C.

param(
    [string]$LogDir = (Join-Path $env:LOCALAPPDATA "Temp\codex-rag")
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Get-Latest {
    Get-ChildItem $LogDir -Filter "task-*.log" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

$shown = $null
$startedAt = Get-Date
Write-Host "Жду задачу Codex в $LogDir ..." -ForegroundColor DarkGray
while ($true) {
    $latest = Get-Latest
    # Показываем только журналы, начатые после запуска наблюдателя,
    # иначе на экран вывалится давно закончившаяся задача.
    if (-not $latest -or $latest.CreationTime -lt $startedAt -or $latest.FullName -eq $shown) {
        Start-Sleep -Seconds 1
        continue
    }
    $shown = $latest.FullName
    Write-Host "`n===== $($latest.Name) =====" -ForegroundColor Cyan

    $stream = New-Object IO.FileStream($shown, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    $reader = New-Object IO.StreamReader($stream, [Text.Encoding]::UTF8)
    try {
        while ($true) {
            $line = $reader.ReadLine()
            if ($null -ne $line) { Write-Host $line; continue }
            # Журнал дочитан. Если появился более новый — переходим к нему.
            $next = Get-Latest
            if ($next -and $next.FullName -ne $shown) { break }
            Start-Sleep -Milliseconds 500
        }
    } finally {
        $reader.Close()
    }
}
