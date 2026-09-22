# Отправить уведомление в Telegram.
#
#   .\scripts\notify.ps1 -Kind done   -Text "Задача 003 принята, готов обсуждать"
#   .\scripts\notify.ps1 -Kind server -Text "Нужен сервер: замер моделей, ~2 ч"
#   .\scripts\notify.ps1 -Kind off    -Text "Сервер больше не нужен, метрики забраны"
#
# Секрет читается из %APPDATA%\rag-notify\telegram.xml (его создаёт
# notify_setup.ps1, запущенный человеком) и расшифровывается только в памяти
# этого процесса. Токен не печатается ни при успехе, ни при ошибке.
# В текст уведомления нельзя класть пароли, ключи, адреса серверов и данные
# проекта: сообщение уходит на серверы Telegram.

param(
    [ValidateSet("done", "server", "off", "blocked", "info")]
    [string]$Kind = "info",
    [Parameter(Mandatory = $true)][string]$Text,
    # Необязательный файл (например, PDF отчёта) — уходит документом с текстом в подписи.
    [string]$File = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$store = Join-Path $env:APPDATA "rag-notify\telegram.xml"
if (-not (Test-Path $store)) {
    Write-Host "Уведомления не настроены: запустите scripts\notify_setup.ps1" -ForegroundColor Yellow
    exit 2
}

$icon = @{ done = "✅"; server = "🖥️"; off = "🔌"; blocked = "⛔"; info = "ℹ️" }[$Kind]
$title = @{
    done    = "Готово, можно обсуждать"
    server  = "Нужен сервер"
    off     = "Сервер можно выключать"
    blocked = "Нужно ваше решение"
    info    = "Сообщение"
}[$Kind]
$body = "$icon $title`n`n$Text`n`n— rag_textbook, $(Get-Date -Format 'dd.MM HH:mm')"
if ($body.Length -gt 3900) { $body = $body.Substring(0, 3900) + "…" }

$secret = Import-Clixml $store
$token = (New-Object PSCredential "bot", $secret.Token).GetNetworkCredential().Password
$chatId = (New-Object PSCredential "chat", $secret.ChatId).GetNetworkCredential().Password

$payload = [Text.Encoding]::UTF8.GetBytes((@{ chat_id = $chatId; text = $body } | ConvertTo-Json -Compress))
try {
    if ($File) {
        # Документ уходит через sendDocument; подпись у Telegram не длиннее 1024 знаков.
        Add-Type -AssemblyName System.Net.Http
        $caption = if ($body.Length -gt 1000) { $body.Substring(0, 1000) + "…" } else { $body }
        $client = New-Object System.Net.Http.HttpClient
        $client.Timeout = [TimeSpan]::FromSeconds(120)
        $form = New-Object System.Net.Http.MultipartFormDataContent
        $form.Add((New-Object System.Net.Http.StringContent($chatId)), "chat_id")
        $form.Add((New-Object System.Net.Http.StringContent($caption, [Text.Encoding]::UTF8)), "caption")
        $bytes = [IO.File]::ReadAllBytes((Resolve-Path $File))
        $part = New-Object System.Net.Http.ByteArrayContent(, $bytes)
        $form.Add($part, "document", [IO.Path]::GetFileName($File))
        $resp = $client.PostAsync("https://api.telegram.org/bot$token/sendDocument", $form).Result
        $client.Dispose()
        if (-not $resp.IsSuccessStatusCode) { throw "HTTP $([int]$resp.StatusCode)" }
    } else {
        [void](Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" `
            -Method Post -Body $payload -ContentType "application/json; charset=utf-8" -TimeoutSec 30)
    }
    Write-Host "Уведомление отправлено ($Kind)." -ForegroundColor Green
    $code = 0
} catch {
    # Сообщение исключения может содержать адрес с токеном — не печатаем его.
    $status = $null
    if ($_.Exception.Response) { $status = [int]$_.Exception.Response.StatusCode }
    Write-Host "Уведомление НЕ отправлено (HTTP $status)." -ForegroundColor Red
    $code = 1
} finally {
    $token = $null; $chatId = $null
}
exit $code
