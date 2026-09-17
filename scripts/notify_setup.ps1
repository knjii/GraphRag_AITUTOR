# Однократная настройка уведомлений в Telegram. Запускает ЧЕЛОВЕК, не агент.
#
#   powershell -ExecutionPolicy Bypass -File scripts\notify_setup.ps1
#
# Порядок:
#   1. В Telegram у @BotFather: /newbot → получить токен.
#   2. Написать своему новому боту любое сообщение (например /start) —
#      без этого бот не может написать вам первым и не узнает ваш chat id.
#   3. Запустить этот скрипт и вставить токен в скрытое поле ввода.
#
# Как хранится секрет. Токен и chat id сохраняются в
# %APPDATA%\rag-notify\telegram.xml через Export-Clixml: Windows шифрует их
# DPAPI ключом вашей учётной записи. Файл бесполезен на другой машине и под
# другим пользователем. Токен нигде не печатается и не попадает в журналы.

#
# Вместо ввода токен можно взять из файла (содержимое не печатается):
#   ... notify_setup.ps1 -TokenFile C:\python\tokens\tg_bot\token_tg.txt

param(
    [string]$TokenFile
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$dir = Join-Path $env:APPDATA "rag-notify"
$store = Join-Path $dir "telegram.xml"

if ($TokenFile) {
    if (-not (Test-Path $TokenFile)) { throw "Нет файла с токеном: $TokenFile" }
    $token = (Get-Content -Raw -Encoding utf8 $TokenFile).Trim().TrimStart([char]0xFEFF)
    $secureToken = ConvertTo-SecureString $token -AsPlainText -Force
} else {
    $secureToken = Read-Host "Токен бота (ввод скрыт)" -AsSecureString
    $token = (New-Object PSCredential "bot", $secureToken).GetNetworkCredential().Password
}
if ($token -notmatch '^\d+:[A-Za-z0-9_-]{30,}$') { throw "Это не похоже на токен бота Telegram" }

function Invoke-Bot([string]$Method, $Body = $null) {
    $uri = "https://api.telegram.org/bot$token/$Method"
    try {
        if ($Body) {
            $bytes = [Text.Encoding]::UTF8.GetBytes(($Body | ConvertTo-Json -Compress))
            return Invoke-RestMethod -Uri $uri -Method Post -Body $bytes -ContentType "application/json; charset=utf-8"
        }
        return Invoke-RestMethod -Uri $uri -Method Get
    } catch {
        # Текст исключения может содержать адрес запроса, а в нём — токен.
        $code = $null
        if ($_.Exception.Response) { $code = [int]$_.Exception.Response.StatusCode }
        throw "Telegram отказал в запросе $Method (HTTP $code). Проверьте токен."
    }
}

$me = Invoke-Bot "getMe"
Write-Host "Бот найден: @$($me.result.username)" -ForegroundColor Green

$updates = Invoke-Bot "getUpdates"
$chats = @($updates.result | ForEach-Object {
    if ($_.message) { $_.message.chat } elseif ($_.my_chat_member) { $_.my_chat_member.chat }
} | Where-Object { $_.type -eq "private" } | Sort-Object id -Unique)

if ($chats.Count -eq 0) {
    throw "Бот ещё не получил от вас сообщений. Напишите ему /start и запустите скрипт снова."
}
if ($chats.Count -gt 1) {
    Write-Host "Боту писали несколько человек:"
    for ($i = 0; $i -lt $chats.Count; $i++) { Write-Host "  [$i] $($chats[$i].first_name) (@$($chats[$i].username))" }
    $chat = $chats[[int](Read-Host "Номер вашего чата")]
} else {
    $chat = $chats[0]
    Write-Host "Чат: $($chat.first_name) (@$($chat.username))"
}

New-Item -ItemType Directory -Force -Path $dir | Out-Null
[pscustomobject]@{
    Token  = $secureToken
    ChatId = (ConvertTo-SecureString ([string]$chat.id) -AsPlainText -Force)
} | Export-Clixml -Path $store

# Файл — только для текущего пользователя.
$acl = Get-Acl $store
$acl.SetAccessRuleProtection($true, $false)
$acl.Access | ForEach-Object { [void]$acl.RemoveAccessRule($_) }
$me_user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$acl.AddAccessRule((New-Object Security.AccessControl.FileSystemAccessRule($me_user, "FullControl", "Allow")))
Set-Acl $store $acl

[void](Invoke-Bot "sendMessage" @{ chat_id = [string]$chat.id; text = "Уведомления rag_textbook настроены." })
$token = $null
Write-Host "Готово. Секрет сохранён в $store (зашифрован DPAPI). Тестовое сообщение отправлено." -ForegroundColor Green
