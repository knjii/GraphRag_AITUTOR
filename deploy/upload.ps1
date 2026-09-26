# Загрузка кода и корпуса на арендованный сервер.
# Запускать на своём компьютере из корня проекта:
#
#   .\deploy\upload.ps1 -ServerIp 1.2.3.4
#   .\deploy\upload.ps1 -ServerIp 1.2.3.4 -WithCaches     # на новый сервер
#
# По умолчанию копируется только код и корпус: артефакты прогонов и виртуальные
# окружения остаются локально, иначе на канал уйдут гигабайты без пользы.
#
# Ключ -WithCaches нужен при переезде на другой сервер. Он переносит кэши
# разбора, обогащения и извлечения — это разница между шестью минутами
# восстановления и полутора часами. Замер: сборка графа по всей книге занимает
# 72 минуты, а с кэшем извлечения — 4 минуты, потому что заново считается
# только проход по связям между фрагментами.

param(
    [Parameter(Mandatory = $true)][string]$ServerIp,
    [string]$User = "root",
    [int]$Port = 22,
    [string]$KeyPath = "$env:USERPROFILE\.ssh\intelion_ed25519",
    [string]$RemoteDir = "~/rag_textbook",
    # Перенести кэши и эталонный набор. Осмысленно при переезде на новый сервер.
    [switch]$WithCaches,
    # Для первого прогона берём один учебник — тот же, на котором снят прежний
    # baseline в 14.5 часа. Иначе ускорение не с чем будет сравнивать.
    [string[]]$Pdfs = @(
        "documents\pdf_docs\Dayzenrot_Feyzal_On_Matematika_v_mashinnom_obuchen_241126_230954.pdf"
    ),
    # Откуда брать данные (кэши, корпус, библиотеку, эпизоды RL). Код ветки
    # agents/dev лежит в отдельном рабочем каталоге, а данные — в основном:
    #   .\deploy\upload.ps1 -ServerIp ... -WithCaches -WithLibrary -DataRoot C:\python\rag_textbook
    [string]$DataRoot = ".",
    # Библиотека книг (documents\library) и эпизоды RL (artifacts\rl) —
    # для дня аренды docs/SERVER-DAY-1.md.
    [switch]$WithLibrary
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $KeyPath)) {
    throw "SSH-ключ не найден: $KeyPath. Создайте его: ssh-keygen -t ed25519 -C `"rag-textbook`" -f $KeyPath"
}

# Порт задаётся разными ключами: у ssh это -p, у scp -p означает «сохранить
# время файла», а порт — -P. Общий массив аргументов здесь использовать нельзя.
# Keep-alive: без него длинная передача рвалась (Broken pipe).
$keepAlive = @("-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=8")
$sshArgs = @("-i", $KeyPath, "-p", $Port) + $keepAlive
$scpArgs = @("-i", $KeyPath, "-P", $Port) + $keepAlive
$target  = "${User}@${ServerIp}"

function Invoke-Remote([string]$Command) {
    & ssh @sshArgs $target $Command
    if ($LASTEXITCODE -ne 0) { throw "Команда на сервере завершилась с ошибкой: $Command" }
}

# Большой каталог — одним архивом и с повтором. scp -r по тысячам мелких
# файлов рвался посреди разбора (Broken pipe, 2026-09-22), а повтор
# с начала стоил бы всей передачи заново.
function Copy-DirArchive([string]$Local, [string]$RemoteParent) {
    $name = Split-Path $Local -Leaf
    $archive = Join-Path $env:TEMP "rag-upload-$name.tar"
    Remove-Item $archive -ErrorAction SilentlyContinue
    & tar -cf $archive -C (Split-Path $Local -Parent) $name
    if ($LASTEXITCODE -ne 0) { throw "Не удалось упаковать $Local" }
    $remoteArchive = "/tmp/rag-upload-$name.tar"
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        # В PowerShell 5.1 сообщение scp в stderr при Stop — исключение,
        # и до повтора дело не дошло бы.
        $saved = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        & scp @scpArgs -q $archive "${target}:$remoteArchive" 2>&1 | Out-Host
        $code = $LASTEXITCODE; $ErrorActionPreference = $saved
        if ($code -eq 0) { break }
        Write-Host "    обрыв передачи $name, попытка $attempt из 3" -ForegroundColor DarkYellow
        if ($attempt -eq 3) { throw "Не удалось скопировать $Local" }
        Start-Sleep -Seconds 10
    }
    Invoke-Remote "mkdir -p $RemoteParent && tar -xf $remoteArchive -C $RemoteParent && rm -f $remoteArchive"
    Remove-Item $archive -ErrorAction SilentlyContinue
}

Write-Host "`n==> Проверяю связь с сервером" -ForegroundColor Cyan
Invoke-Remote "echo OK && nvidia-smi --query-gpu=name --format=csv,noheader"

Write-Host "`n==> Создаю каталоги" -ForegroundColor Cyan
Invoke-Remote "mkdir -p $RemoteDir/documents/pdf_docs $RemoteDir/deploy"

Write-Host "`n==> Копирую код" -ForegroundColor Cyan
# capture обязателен: там лежат слепки поиска, по которым идут замеры
# с замороженным контекстом. Без них блок сравнения генераторов падает
# на первой строке, уже на оплаченной карте.
# scripts — офлайн-разборы; они мелкие, а искать их потом дороже.
$codePaths = @(
    "rag_textbook", "tests", "docker", "deploy", "docs", "scripts", "capture",
    "tasks", "pyproject.toml", ".env.example", "README.md"
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

if ($WithCaches) {
    Write-Host "`n==> Копирую кэши и эталонный набор" -ForegroundColor Cyan
    # Порядок важен только для наглядности. Каждый каталог самодостаточен:
    #   parsed    — результат MinerU и готовые чанки, снимает стадию разбора;
    #   cache     — описания иллюстраций и извлечённые сущности со связями;
    #   manifests — отметки о выполненных стадиях;
    #   (эталоны едут вместе с кодом ниже — из каталога ветки).
    Invoke-Remote "mkdir -p $RemoteDir/artifacts $RemoteDir/evaluation"
    $cachePaths = @(
        @{ Local = "artifacts\parsed";     Remote = "artifacts" },
        @{ Local = "artifacts\cache";      Remote = "artifacts" },
        @{ Local = "artifacts\manifests";  Remote = "artifacts" }
    )
    foreach ($item in $cachePaths) {
        $item.Local = Join-Path $DataRoot $item.Local
        if (-not (Test-Path $item.Local)) {
            Write-Host "    пропускаю отсутствующий $($item.Local)" -ForegroundColor DarkYellow
            continue
        }
        $sizeMb = [math]::Round(
            (Get-ChildItem $item.Local -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1
        )
        Write-Host "    $($item.Local) ($sizeMb МБ)"
        Copy-DirArchive $item.Local "${RemoteDir}/$($item.Remote)"
    }
    Write-Host "    после развёртывания восстановите индекс двумя командами:" -ForegroundColor DarkGray
    Write-Host "      rag-textbook ingest --stages parse,chunk,embed --force" -ForegroundColor DarkGray
    Write-Host "      rag-textbook ingest --stages graph --force" -ForegroundColor DarkGray
}

# Эталоны, манифест библиотеки и ручные сверки награды — данные эксперимента,
# их место в evaluation. Берутся из рабочего каталога ветки, а не из -DataRoot:
# они отслеживаются git, а перенесённый эталон (goldset-r2) есть только
# в ветке — из основного каталога уехал бы прежний, с номерами старой нарезки.
Invoke-Remote "mkdir -p $RemoteDir/evaluation"
foreach ($path in @("evaluation\goldsets", "evaluation\library", "evaluation\reward_checks")) {
    if (Test-Path $path) {
        & scp @scpArgs -r -q $path "${target}:${RemoteDir}/evaluation/"
        if ($LASTEXITCODE -ne 0) { throw "Не удалось скопировать $path" }
    }
}
# Нарезка MML, на которую перенесён goldset-r2 (1247 фрагментов, отпечаток
# в goldset-r2.migration.json). Сервер режет сам и сверяет отпечаток;
# при расхождении deploy/day2.sh ставит этот файл, а не останавливает день.
if (Test-Path "artifacts\goldset-r2") {
    Invoke-Remote "mkdir -p $RemoteDir/artifacts"
    & scp @scpArgs -r -q "artifacts\goldset-r2" "${target}:${RemoteDir}/artifacts/"
    if ($LASTEXITCODE -ne 0) { throw "Не удалось скопировать artifacts\goldset-r2" }
}

if ($WithLibrary) {
    Write-Host "`n==> Копирую библиотеку и эпизоды RL" -ForegroundColor Cyan
    Invoke-Remote "mkdir -p $RemoteDir/documents $RemoteDir/artifacts"
    $libraryPaths = @(
        @{ Local = "documents\library"; Remote = "documents" },
        @{ Local = "artifacts\rl";      Remote = "artifacts" }
    )
    foreach ($item in $libraryPaths) {
        $local = Join-Path $DataRoot $item.Local
        if (-not (Test-Path $local)) { throw "Нет ${local}: см. docs/SERVER-DAY-1.md, раздел «До аренды»" }
        $sizeMb = [math]::Round((Get-ChildItem $local -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
        Write-Host "    $local ($sizeMb МБ)"
        Copy-DirArchive $local "${RemoteDir}/$($item.Remote)"
    }
    # Контрольные суммы: книга на сервере должна быть той же, что проверена здесь.
    Invoke-Remote "cd $RemoteDir/documents/library && sha256sum -c --quiet SHA256SUMS"
}

Write-Host "`n==> Копирую корпус" -ForegroundColor Cyan
foreach ($pdf in $Pdfs) {
    $pdf = Join-Path $DataRoot $pdf
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
