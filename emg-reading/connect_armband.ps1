# Joins the armband's WiFi and gets it streaming to this PC (Windows desktop only, the Mac needs none of this).
# The desktop's WiFi card never gets a DHCP address from the armband, so it needs a static IP. But after every
# armband power-up the band only streams to a device that asked it for an address since it booted, so the adapter
# goes to DHCP for a while and then back to static (SETUP.md). Changing the address needs admin rights: run
# without them, the script asks once through the UAC prompt.
#
#   powershell -ExecutionPolicy Bypass -File connect_armband.ps1
param(
    [string]$Ssid = "Mindrove_ARB_66bb04",
    [string]$Password = "#mindrove",
    [string]$Adapter = "WiFi",
    [string]$StaticIp = "192.168.4.2",
    [string]$BandIp = "192.168.4.1",
    [int]$DhcpSeconds = 20,
    # Progress goes here instead of the console, so the elevated copy can pass it back
    [string]$Log
)

$started = Get-Date

function Say($text) {
    $line = "[{0:mm\:ss}] {1}" -f ((Get-Date) - $started), $text
    if ($Log) { Add-Content -Path $Log -Value $line -Encoding utf8 } else { Write-Output $line }
}

# Called netsh.exe explicitly: a function named like the program would call itself
function Invoke-Netsh {
    $out = & netsh.exe @args 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) { throw "netsh $($args -join ' ') failed: $($out.Trim())" }
    $out
}

function Wifi-Field($name) {
    $m = [regex]::Match((& netsh.exe wlan show interfaces | Out-String), "(?m)^\s*$name\s*:\s*(.+?)\s*$")
    if ($m.Success) { $m.Groups[1].Value } else { "" }
}

function Address {
    $a = Get-NetIPAddress -InterfaceAlias $Adapter -AddressFamily IPv4 -ErrorAction SilentlyContinue
    if ($a) { ($a | ForEach-Object { "$($_.IPAddress) ($($_.PrefixOrigin))" }) -join ", " } else { "none" }
}

$identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Say "Changing the adapter's address needs admin rights: accept the Windows prompt on the PC's screen"
    $out = if ($Log) { $Log } else { [IO.Path]::GetTempFileName() }
    $argv = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"", "-Ssid", "`"$Ssid`"",
        "-Password", "`"$Password`"", "-Adapter", "`"$Adapter`"", "-StaticIp", $StaticIp, "-BandIp", $BandIp,
        "-DhcpSeconds", $DhcpSeconds, "-Log", "`"$out`"")
    try {
        $p = Start-Process powershell -Verb RunAs -ArgumentList $argv -WindowStyle Hidden -Wait -PassThru
    } catch {
        Say "Admin rights refused at the prompt, nothing was changed"
        exit 2
    }
    if (-not $Log) { Get-Content $out; Remove-Item $out }
    exit $p.ExitCode
}

$profileFile = Join-Path $env:TEMP "armband-wifi.xml"
try {
    Say "Running with admin rights"
    $nic = Get-NetAdapter -Name $Adapter -ErrorAction SilentlyContinue
    if (-not $nic) { throw "no network adapter called '$Adapter' (Get-NetAdapter lists them)" }
    Say "Adapter $Adapter ($($nic.InterfaceDescription)): $($nic.Status), address $(Address)"

    Say "Step 1/5: forgetting the saved $Ssid network"
    & netsh.exe wlan delete profile name="$Ssid" interface="$Adapter" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Say "  saved profile deleted" } else { Say "  none was saved" }

    Say "Step 2/5: adding $Ssid again (WPA2-Personal, AES, password from the script)"
    $name = [Security.SecurityElement]::Escape($Ssid)
    $key = [Security.SecurityElement]::Escape($Password)
    @"
<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>$name</name>
  <SSIDConfig><SSID><name>$name</name></SSID></SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>auto</connectionMode>
  <MSM><security>
    <authEncryption><authentication>WPA2PSK</authentication><encryption>AES</encryption><useOneX>false</useOneX></authEncryption>
    <sharedKey><keyType>passPhrase</keyType><protected>false</protected><keyMaterial>$key</keyMaterial></sharedKey>
  </security></MSM>
</WLANProfile>
"@ | Set-Content -Path $profileFile -Encoding utf8
    Invoke-Netsh wlan add profile filename="$profileFile" interface="$Adapter" | Out-Null
    Say "  profile added"

    Say "Step 3/5: connecting to $Ssid"
    Invoke-Netsh wlan connect name="$Ssid" interface="$Adapter" | Out-Null
    $joined = $false
    for ($i = 1; $i -le 30 -and -not $joined; $i++) {
        Start-Sleep -Milliseconds 500
        $joined = (Wifi-Field "State") -eq "connected" -and (Wifi-Field "SSID") -eq $Ssid
        if (-not $joined -and $i % 4 -eq 0) { Say "  waiting, state: $(Wifi-Field 'State')" }
    }
    if (-not $joined) { throw "couldn't join $Ssid in 15 s (state: $(Wifi-Field 'State')), is the armband on?" }
    Say "  joined $Ssid, signal $(Wifi-Field 'Signal'), address $(Address)"

    Say "Step 4/5: adapter to DHCP for $DhcpSeconds s, so the armband sees an address request"
    Invoke-Netsh interface ip set address name="$Adapter" source=dhcp | Out-Null
    for ($left = $DhcpSeconds; $left -gt 0; $left -= 5) {
        Start-Sleep -Seconds ([Math]::Min(5, $left))
        Say "  $([Math]::Max(0, $left - 5)) s left, address $(Address) (169.254.x.x is expected on this PC)"
    }

    Say "Step 5/5: static $StaticIp back on"
    Invoke-Netsh interface ip set address name="$Adapter" static $StaticIp 255.255.255.0 | Out-Null
    Start-Sleep -Seconds 2
    Say "  address $(Address)"

    Say "Pinging the armband at $BandIp"
    $reached = $false
    for ($i = 1; $i -le 10 -and -not $reached; $i++) {
        $reached = Test-Connection $BandIp -Count 1 -Quiet
        if (-not $reached) { Say "  no answer (try $i of 10)"; Start-Sleep -Seconds 1 }
    }
    if (-not $reached) { throw "static IP set but the armband at $BandIp doesn't answer" }
    Say "  armband answers"
    exit 0
} catch {
    Say "FAILED: $($_.Exception.Message)"
    exit 1
} finally {
    Remove-Item $profileFile -ErrorAction SilentlyContinue
}
