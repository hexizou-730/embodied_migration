param(
    [string]$VMName = "embodied-ubuntu-dev",
    [string]$VmsRoot = "F:\Embodied\vms",
    [string]$IsoDir = "F:\Embodied\iso",
    [string]$IsoUrl = "https://releases.ubuntu.com/24.04/ubuntu-24.04.4-desktop-amd64.iso",
    [string]$Sha256SumsUrl = "https://releases.ubuntu.com/24.04/SHA256SUMS",
    [int]$MemoryMB = 8192,
    [int]$Cpus = 6,
    [int]$DiskGB = 120,
    [switch]$SkipIsoDownload,
    [switch]$InstallOrRepairVirtualBox,
    [switch]$StartAfterCreate
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "[vm-setup] $Message" -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host ""
    Write-Host "[vm-setup:warning] $Message" -ForegroundColor Yellow
}

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Find-VBoxManage {
    $candidates = @(
        "$env:ProgramFiles\Oracle\VirtualBox\VBoxManage.exe",
        "${env:ProgramFiles(x86)}\Oracle\VirtualBox\VBoxManage.exe"
    )

    $registryRoots = @(
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*"
    )
    foreach ($root in $registryRoots) {
        Get-ItemProperty $root -ErrorAction SilentlyContinue |
            Where-Object { $_.DisplayName -like "*VirtualBox*" -and $_.InstallLocation } |
            ForEach-Object {
                $candidates += (Join-Path $_.InstallLocation "VBoxManage.exe")
            }
    }

    $cmd = Get-Command VBoxManage.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        $candidates += $cmd.Source
    }

    foreach ($path in $candidates | Select-Object -Unique) {
        if ($path -and (Test-Path -LiteralPath $path)) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    return $null
}

function Install-OrRepair-VirtualBox {
    Write-Step "Installing or repairing Oracle VirtualBox with winget."
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget.exe was not found. Install VirtualBox manually from https://www.virtualbox.org/."
    }

    if (-not (Test-IsAdmin)) {
        Write-Warn "This shell is not elevated. winget may open a UAC prompt or fail. Run PowerShell as Administrator if that happens."
    }

    $installed = (& $winget.Source list --id Oracle.VirtualBox --accept-source-agreements 2>$null) -join "`n"
    if ($installed -match "Oracle\.VirtualBox") {
        & $winget.Source upgrade --id Oracle.VirtualBox -e `
            --accept-package-agreements --accept-source-agreements
    }
    else {
        & $winget.Source install --id Oracle.VirtualBox -e `
            --accept-package-agreements --accept-source-agreements
    }
}

function Download-UbuntuIso {
    param(
        [string]$Url,
        [string]$DestinationDir,
        [string]$SumsUrl
    )

    New-Item -ItemType Directory -Force -Path $DestinationDir | Out-Null
    $fileName = Split-Path -Leaf ([Uri]$Url).AbsolutePath
    $isoPath = Join-Path $DestinationDir $fileName

    if (-not (Test-Path -LiteralPath $isoPath)) {
        Write-Step "Downloading Ubuntu ISO to $isoPath"
        Write-Host "Source: $Url"
        Invoke-WebRequest -Uri $Url -OutFile $isoPath
    }
    else {
        Write-Step "Ubuntu ISO already exists: $isoPath"
    }

    Write-Step "Checking ISO SHA256 against official SHA256SUMS."
    $sums = Invoke-WebRequest -Uri $SumsUrl -UseBasicParsing
    $expectedLine = ($sums.Content -split "`n") | Where-Object { $_ -match [regex]::Escape($fileName) } | Select-Object -First 1
    if (-not $expectedLine) {
        Write-Warn "Could not find $fileName in SHA256SUMS. Skipping checksum verification."
        return $isoPath
    }

    $expected = ($expectedLine.Trim() -split "\s+")[0].ToLowerInvariant()
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $isoPath).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "ISO checksum mismatch. Expected $expected, got $actual. Delete the ISO and run again."
    }
    Write-Host "SHA256 OK: $actual"
    return $isoPath
}

if ($InstallOrRepairVirtualBox) {
    Install-OrRepair-VirtualBox
}

$vbox = Find-VBoxManage
if (-not $vbox) {
    throw @"
VBoxManage.exe was not found.

Run this first from PowerShell:
  winget install --id Oracle.VirtualBox -e --accept-package-agreements --accept-source-agreements

If winget says VirtualBox is installed but this script still cannot find VBoxManage,
repair/upgrade it:
  .\scripts\windows_prepare_virtualbox_vm.ps1 -InstallOrRepairVirtualBox
"@
}

Write-Step "Using VBoxManage: $vbox"
& $vbox --version

New-Item -ItemType Directory -Force -Path $VmsRoot | Out-Null

if ($SkipIsoDownload) {
    $isoCandidates = Get-ChildItem -LiteralPath $IsoDir -Filter "ubuntu-24.04*-desktop-amd64.iso" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending
    if (-not $isoCandidates) {
        throw "No Ubuntu desktop ISO found in $IsoDir. Remove -SkipIsoDownload or place an ISO there."
    }
    $isoPath = $isoCandidates[0].FullName
}
else {
    $isoPath = Download-UbuntuIso -Url $IsoUrl -DestinationDir $IsoDir -SumsUrl $Sha256SumsUrl
}

$existing = (& $vbox list vms) -join "`n"
if ($existing -match ('"' + [regex]::Escape($VMName) + '"')) {
    Write-Warn "VM already exists: $VMName"
    Write-Host "Open it in VirtualBox, or delete/rename it before recreating."
    exit 0
}

$vmDir = Join-Path $VmsRoot $VMName
$diskPath = Join-Path $vmDir "$VMName.vdi"
$diskMB = $DiskGB * 1024

Write-Step "Creating VM: $VMName"
& $vbox createvm --name $VMName --ostype Ubuntu_64 --basefolder $VmsRoot --register

& $vbox modifyvm $VMName `
    --memory $MemoryMB `
    --cpus $Cpus `
    --vram 128 `
    --graphicscontroller vmsvga `
    --accelerate3d on `
    --nic1 nat `
    --audio-driver default `
    --clipboard-mode bidirectional `
    --draganddrop bidirectional `
    --boot1 dvd `
    --boot2 disk `
    --boot3 none `
    --boot4 none

& $vbox storagectl $VMName --name "SATA Controller" --add sata --controller IntelAhci --bootable on
& $vbox createmedium disk --filename $diskPath --size $diskMB --format VDI --variant Standard
& $vbox storageattach $VMName --storagectl "SATA Controller" --port 0 --device 0 --type hdd --medium $diskPath
& $vbox storageattach $VMName --storagectl "SATA Controller" --port 1 --device 0 --type dvddrive --medium $isoPath

$sharedRepo = "F:\Embodied\embodied_migration"
if (Test-Path -LiteralPath $sharedRepo) {
    & $vbox sharedfolder add $VMName `
        --name embodied_migration_win `
        --hostpath $sharedRepo `
        --automount `
        --readonly
}

Write-Step "VM created."
Write-Host "Name:       $VMName"
Write-Host "Memory:     $MemoryMB MB"
Write-Host "CPUs:       $Cpus"
Write-Host "Disk:       $DiskGB GB"
Write-Host "ISO:        $isoPath"
Write-Host "VM folder:  $vmDir"
Write-Host ""
Write-Host "Next:"
Write-Host "  1. Start the VM."
Write-Host "  2. Install Ubuntu Desktop."
Write-Host "  3. In Ubuntu, run scripts/ubuntu_vm_bootstrap_dev.sh or the commands in docs/VM_UBUNTU_SETUP_CN.md."

if ($StartAfterCreate) {
    Write-Step "Starting VM with GUI."
    & $vbox startvm $VMName --type gui
}
