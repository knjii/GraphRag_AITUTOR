# Забрать результаты прогона с сервера на свой компьютер.
#
#   .\deploy\download.ps1 -ServerIp 1.2.3.4
#
# Смысл в том, чтобы разбор результатов вести локально и бесплатно,
# а не платить 45.70 ₽/час за просмотр JSON-файлов.

param(
    [Parameter(Mandatory = $true)][string]$ServerIp,
    [string]$User = "root",
    [int]$Port = 22,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\intelion_ed25519",
    [string]$RemoteDir = "~/rag_textbook",
    [string]$LocalDir = "artifacts\from-server"
)

$ErrorActionPreference = "Stop"
$sshArgs = @("-i", $KeyPath, "-p", $Port)
$target  = "${User}@${ServerIp}"

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

Write-Host "`n==> Ищу архивы прогонов на сервере" -ForegroundColor Cyan
$archives = & ssh @sshArgs $target "ls -1 $RemoteDir/artifacts/pilot_*.tar.gz 2>/dev/null"
if (-not $archives) {
    Write-Host "    Архивов не найдено. Прогон ещё не завершался?" -ForegroundColor Yellow
    exit 0
}

foreach ($archive in $archives) {
    $name = Split-Path $archive -Leaf
    Write-Host "    забираю $name"
    & scp @sshArgs -q "${target}:${archive}" "$LocalDir\"
    if ($LASTEXITCODE -ne 0) { throw "Не удалось скачать $name" }
}

Write-Host "`n==> Скачано в $LocalDir" -ForegroundColor Green
Get-ChildItem $LocalDir | Format-Table Name, @{N = "Размер, МБ"; E = { [math]::Round($_.Length / 1MB, 2) } }

Write-Host @"

Распаковать:
    tar -xzf $LocalDir\<имя>.tar.gz -C $LocalDir

Смотреть в первую очередь:
    summary.json                     тайминги стадий и утилизация GPU
    metrics/retrieval_eval_*.json    метрики поиска
    metrics/indexing_*.json          отчёт индексации по документам
    pilot.log                        полный лог прогона

"@ -ForegroundColor Cyan
