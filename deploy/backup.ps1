# Забрать с сервера всё, что нельзя восстановить бесплатно.
#
#   .\deploy\backup.ps1 -ServerIp 185.182.108.41
#   .\deploy\backup.ps1 -ServerIp 185.182.108.41 -WithCaches
#
# Зачем. 2026-09-09 машину пересоздали, и вместе с ней исчезли 73 ГБ весов,
# индекс Qdrant, граф Neo4j и кэши. Восстановление заняло час — и оно вообще
# оказалось возможным только потому, что разбор PDF и кэш извлечения лежали
# локально. Единственное, чего локально НЕ БЫЛО, — эталонный набор на 388
# вопросов: рабочая копия в репозитории отстала на 163 вопроса, и первый же
# замер после восстановления посчитался не по тому набору.
#
# Что забираем и почему:
#   metrics   — результаты замеров, их не воспроизвести без аренды;
#   goldsets  — набор и вердикты ручной проверки, растут на сервере;
#   cache     — извлечённые связи, 72 минуты работы модели;
#   parsed    — разбор PDF, пять минут MinerU на книгу плюс недетерминизм.
#
# Чего НЕ забираем: веса моделей. Это публичные файлы на 73 ГБ, сервер
# качает их за минуты по своему каналу, домашний — за часы.

param(
    [Parameter(Mandatory = $true)][string]$ServerIp,
    [string]$User = "root",
    [int]$Port = 22,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\intelion_ed25519",
    [string]$RemoteDir = "rag_textbook",
    # Кэши разбора и извлечения. Меняются редко, весят 300 МБ.
    [switch]$WithCaches
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $KeyPath)) { throw "SSH-ключ не найден: $KeyPath" }

$target = "${User}@${ServerIp}"
$scpArgs = @("-i", $KeyPath, "-P", $Port)

function Get-Remote {
    param([string]$Remote, [string]$Local, [string]$Title)
    Write-Host "    $Title" -ForegroundColor Gray
    New-Item -ItemType Directory -Force -Path $Local | Out-Null
    & scp @scpArgs -r -q "${target}:${RemoteDir}/${Remote}" $Local
    if ($LASTEXITCODE -ne 0) { Write-Host "      пропущено: на сервере нет" -ForegroundColor DarkYellow }
}

Write-Host "`n==> Забираю результаты замеров" -ForegroundColor Cyan
$stamp = Get-Date -Format "yyyyMMdd"
Get-Remote "artifacts/metrics/*" "capture/server-$stamp" "метрики"

Write-Host "`n==> Забираю эталонный набор и вердикты" -ForegroundColor Cyan
# Набор растёт и правится на сервере. Именно его отставание стоило нам
# ячейки, посчитанной по 163 вопросам вместо 388.
Get-Remote "evaluation/goldsets/*" "evaluation/goldsets" "набор и вердикты"

if ($WithCaches) {
    Write-Host "`n==> Забираю кэши" -ForegroundColor Cyan
    Get-Remote "artifacts/cache" "artifacts" "кэш извлечения связей"
    Get-Remote "artifacts/parsed" "artifacts" "разбор PDF"
}

Write-Host @"

===============================================================================
  Выгрузка завершена.

  Дальше — обязательно закоммитить набор, если он изменился:
      git status evaluation/goldsets/
      git add evaluation/goldsets/ && git commit -m "Обновлён эталонный набор"

  Веса моделей намеренно не забираются: 73 ГБ публичных файлов, сервер
  скачивает их сам за минуты командой deploy/models-fetch.sh
===============================================================================
"@ -ForegroundColor Green
