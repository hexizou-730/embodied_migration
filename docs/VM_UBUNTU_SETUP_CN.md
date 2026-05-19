# 虚拟机 Ubuntu 前期实验环境

这份文档用于当前阶段：先在 Windows 上用 Ubuntu 虚拟机完成代码框架、LLM 闭环、日志和静态 baseline 调试。正式 ManiSkill GPU/GUI 实验后续再迁到原生 Ubuntu。

## 你的当前主机情况

已检查到：

- Windows 10 Home；
- CPU: AMD Ryzen 7 5800H，16 logical processors；
- 内存约 16 GB；
- F 盘剩余约 686 GB，适合放虚拟机；
- BIOS 虚拟化已开启；
- winget 可用；
- 系统登记过 Oracle VirtualBox 7.2.4，但当前没有找到 `VBoxManage.exe`，可能需要修复或升级 VirtualBox。

## 方案选择

推荐先用 VirtualBox：

- Windows Home 不适合依赖完整 Hyper-V 管理器；
- VirtualBox 足够用于 Python/LLM/日志/静态检查；
- 虚拟机数据后续可通过 Git 和 `rsync` 迁到原生 Ubuntu。

虚拟机配置建议：

- Ubuntu Desktop 24.04 LTS；
- CPU: 6 cores；
- RAM: 8192 MB；
- Disk: 120 GB；
- Graphics: VMSVGA + 128 MB VRAM + 3D acceleration；
- Network: NAT；
- Shared clipboard: bidirectional。

## 1. 修复或安装 VirtualBox

先在 PowerShell 中检查：

```powershell
Test-Path "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe"
```

如果返回 `False`，用管理员 PowerShell 执行：

```powershell
winget install --id Oracle.VirtualBox -e --accept-package-agreements --accept-source-agreements
```

如果 winget 认为已经安装，则执行升级/修复：

```powershell
winget upgrade --id Oracle.VirtualBox -e --accept-package-agreements --accept-source-agreements
```

安装或升级后重启 Windows，再检查：

```powershell
Test-Path "C:\Program Files\Oracle\VirtualBox\VBoxManage.exe"
```

## 2. 自动下载 Ubuntu ISO 并创建 VM

在 Windows PowerShell 中进入仓库：

```powershell
cd F:\Embodied\embodied_migration
```

执行：

```powershell
.\scripts\windows_prepare_virtualbox_vm.ps1 -StartAfterCreate
```

这个脚本会：

- 下载 Ubuntu 24.04.4 Desktop ISO 到 `F:\Embodied\iso`；
- 用官方 `SHA256SUMS` 校验 ISO；
- 创建 `embodied-ubuntu-dev` 虚拟机；
- 把 VM 放到 `F:\Embodied\vms`；
- 配置 8 GB 内存、6 核 CPU、120 GB 虚拟硬盘；
- 挂载 Ubuntu ISO 并启动 VM；
- 只读挂载 Windows 侧仓库为共享文件夹 `embodied_migration_win`。

如果你已经手动下载了 ISO，可以放到 `F:\Embodied\iso`，然后运行：

```powershell
.\scripts\windows_prepare_virtualbox_vm.ps1 -SkipIsoDownload -StartAfterCreate
```

## 3. 安装 Ubuntu

VM 启动后，在 Ubuntu 安装器中选择：

- Language: English 或中文均可；
- Keyboard: Chinese/English 按你习惯；
- Installation type: Erase disk and install Ubuntu。

这里的 “Erase disk” 只会清空虚拟机里的 120 GB 虚拟硬盘，不会清空你的 Windows 硬盘。

安装完成后重启 VM。如果它又进入安装界面，在 VirtualBox 里弹出 Ubuntu ISO，或把启动顺序改成硬盘优先。

## 4. 虚拟机内安装项目开发环境

进入 Ubuntu 桌面后打开 Terminal：

```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

重启后执行：

```bash
mkdir -p ~/Embodied
cd ~/Embodied
git clone https://github.com/hexizou-730/embodied_migration.git
cd embodied_migration
bash scripts/ubuntu_vm_bootstrap_dev.sh
```

这个脚本会跳过 NVIDIA driver，只安装前期开发需要的依赖。

## 5. 验证

```bash
cd ~/Embodied/embodied_migration
conda activate em-ms
python -m compileall -q .
python - <<'PY'
from capabilities.capability_card import CapabilityCard
from lmp.failure_report import FailureReport
from lmp.extractor import extract_code_or_text
print(CapabilityCard().to_prompt_section().splitlines()[0])
print(FailureReport(task_name="smoke", instruction="test", robot_name="panda").to_prompt_section().splitlines()[0])
print(extract_code_or_text("```python\nret_val = 1\n```"))
PY
```

如果这些通过，就说明虚拟机已经能做前期开发了。

## 6. 现阶段不要在虚拟机里死磕的内容

不要把时间花在：

- NVIDIA driver；
- `nvidia-smi`；
- Vulkan 真 GPU 渲染；
- ManiSkill viewer 稳定性；
- 大规模实验矩阵。

这些等迁到原生 Ubuntu 后再做。

## 7. 如果 VirtualBox 很慢

如果状态栏出现类似绿色 turtle，通常是 Windows Hyper-V/VBS 正在接管虚拟化。前期开发仍可能能用；如果明显卡顿，再考虑关闭：

- Windows Hypervisor Platform；
- Virtual Machine Platform；
- Windows Subsystem for Linux；
- Core Isolation / Memory Integrity。

这些改动需要管理员权限和重启，不建议一开始就折腾。先确认虚拟机能跑 Python 和项目代码。

