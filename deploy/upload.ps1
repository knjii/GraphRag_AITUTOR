# Загрузка кода и корпуса на арендованный сервер.
# Запускать на своём компьютере из корня проекта:
#
#   .\deploy\upload.ps1 -ServerIp 1.2.3.4
#
# Копируется только необходимое: артефакты прогонов, кэши и виртуальные окружения
# остаются локально — иначе на канал уйдут гигабайты без пользы.

param(
    [Parameter(Mandatory = $true)][string]$ServerIp,
    [string]$User = "root",
    [int]$Port = 22,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\intelion_ed25519",
    [string]$RemoteDir = "~/rag_textbook",
    # Для первого прогона берём один учебник — тот же, на котором снят прежний
    # baseline в 14.5 часа. Иначе ускорение не с чем будет сравнивать.
    [string[]]$Pdfs = @(
        "documents\pdf_docs\Dayzenrot_Feyzal_On_Matematika_v_mashinnom_obuchen_241126_230954.pdf"
    )
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $KeyPath)) {
    throw "SSH-ключ не найден: $KeyPath. Создайте его: ssh-keygen -t ed25519 -C `"rag-textbook`" -f $KeyPath"
}

# Порт задаётся разными ключами: у ssh это -p, у scp -p означает «сохранить
# время файла», а порт — -P. Общий массив аргументов здесь использовать нельзя.
$sshArgs = @("-i", $KeyPath, "-p", $Port)
$scpArgs = @("-i", $KeyPath, "-P", $Port)
$target  = "${User}@${ServerIp}"

function Invoke-Remote([string]$Command) {
    & ssh @sshArgs $target $Command
    if ($LASTEXITCODE -ne 0) { throw "Команда на сервере завершилась с ошибкой: $Command" }
}

Write-Host "`n==> Проверяю связь с сервером" -ForegroundColor Cyan
Invoke-Remote "echo OK && nvidia-smi --query-gpu=name --format=csv,noheader"

Write-Host "`n==> Создаю каталоги" -ForegroundColor Cyan
Invoke-Remote "mkdir -p $RemoteDir/documents/pdf_docs $RemoteDir/deploy"

Write-Host "`n==> Копирую код" -ForegroundColor Cyan
$codePaths = @(
    "rag_textbook", "tests", "docker", "deploy", "docs",
    "pyproject.toml", ".env.example", "README.md"
)
foreach ($path in $codePaths) {
    if (-not (Test-Path $path)) {
        Write-Host "    пропускаю отсутствующий $path" -ForegroundColor DarkYellow
        continue
    }
    Write-Host "    $path"
    & scp @scpArgs -r -q $path "${target}:${RemoteDir}/"
    if ($LASTEXITCODE -ne 0) { throw "Не удалось скопировать $path" }
}

Write-Host "`n==> Копирую корпус" -ForegroundColor Cyan
foreach ($pdf in $Pdfs) {
    if (-not (Test-Path $pdf)) {
        Write-Host "    ФАЙЛ НЕ НАЙДЕН: $pdf" -ForegroundColor Red
        continue
    }
    $sizeMb = [math]::Round((Get-Item $pdf).Length / 1MB, 1)
    Write-Host "    $(Split-Path $pdf -Leaf) ($sizeMb МБ)"
    & scp @scpArgs -q $pdf "${target}:${RemoteDir}/documents/pdf_docs/"
    if ($LASTEXITCODE -ne 0) { throw "Не удалось скопировать $pdf" }
}

Write-Host "`n==> Делаю скрипты исполняемыми" -ForegroundColor Cyan
Invoke-Remote "chmod +x $RemoteDir/deploy/*.sh"

Write-Host "`n==> Загруженный корпус" -ForegroundColor Cyan
Invoke-Remote "ls -lh $RemoteDir/documents/pdf_docs/"

Write-Host @"

===============================================================================
  Загрузка завершена.

  Подключиться к серверу:
      ssh -i $KeyPath -p $Port $target

  Дальше на сервере:
      cd rag_textbook
      bash deploy/bootstrap.sh      # один раз на новой машине, ~40 минут
      bash deploy/services.sh up    # поднять сервисы, ~10 минут
      bash deploy/pilot.sh          # пилотный прогон
===============================================================================

"@ -ForegroundColor Green
