param(
    [string]$Distro = "Ubuntu-24.04",
    [switch]$Install
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "[wsl2-setup] $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host ""
    Write-Host "[wsl2-setup:warning] $Message" -ForegroundColor Yellow
}

function Invoke-WslText {
    param([string[]]$Arguments)
    $bytes = & wsl.exe @Arguments 2>&1 | ForEach-Object {
        [Text.Encoding]::Unicode.GetBytes($_ + "`r`n")
    }
    if (-not $bytes) {
        return ""
    }
    return [Text.Encoding]::Unicode.GetString(($bytes | ForEach-Object { $_ }) -as [byte[]])
}

$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $wsl) {
    throw "wsl.exe was not found. Update Windows or enable Windows Subsystem for Linux first."
}

Write-Step "Current WSL status"
& wsl.exe --status

Write-Step "Installed distributions"
$listOutput = (& wsl.exe --list --verbose 2>&1) -join "`n"
Write-Host $listOutput

if ($listOutput -match [regex]::Escape($Distro)) {
    Write-Step "$Distro is already installed."
    & wsl.exe --set-default-version 2
    & wsl.exe --set-default $Distro
    Write-Host ""
    Write-Host "Open it with:"
    Write-Host "  wsl -d $Distro"
    exit 0
}

if (-not $Install) {
    Write-Warn "$Distro is not installed yet."
    Write-Host ""
    Write-Host "Install it with:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\windows_prepare_wsl2.ps1 -Install"
    Write-Host ""
    Write-Host "Or manually:"
    Write-Host "  wsl --install -d $Distro"
    exit 0
}

Write-Step "Installing $Distro"
Write-Warn "This may require Administrator permission and may ask for a Windows reboot."
& wsl.exe --install -d $Distro

Write-Step "After installation"
Write-Host "If Windows asks for a reboot, reboot first."
Write-Host "Then open Ubuntu and create the Linux username/password."
Write-Host "After that, run inside WSL:"
Write-Host "  mkdir -p ~/Embodied"
Write-Host "  cd ~/Embodied"
Write-Host "  git clone https://github.com/hexizou-730/embodied_migration.git"
Write-Host "  cd embodied_migration"
Write-Host "  bash scripts/wsl2_bootstrap_dev.sh"
