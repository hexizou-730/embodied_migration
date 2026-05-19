# Ubuntu + ManiSkill 环境配置

本项目的正式仿真实验建议在原生 Ubuntu 22.04 / 24.04 上运行，不建议把 WSL 作为主实验平台。当前脚本会完成：

- 安装基础编译工具、Vulkan 运行库和 `ubuntu-drivers` 工具；
- 通过 Ubuntu 官方推荐的 `ubuntu-drivers` 安装 NVIDIA driver；
- 安装 Miniforge/conda；
- 创建 `em-ms` 环境，Python 3.10；
- 安装 `requirements.txt` 和 `requirements-maniskill.txt`；
- 验证 `numpy/openai/torch/gymnasium/mani_skill` 能否导入。

## 0. 如果电脑还没有 Ubuntu

新装建议优先选择 **Ubuntu Desktop 24.04 LTS**。22.04 LTS 也可用，但新装系统时优先用 24.04 LTS，后续驱动和系统包支持会更长。

推荐方案：

- 首选：Windows + Ubuntu 双系统，Ubuntu 安装到独立磁盘空间。
- 备选：安装到一块外置 SSD，便于不改动 Windows 主硬盘分区。
- 可用于前期调试：普通虚拟机中的 Ubuntu。
- 不建议：WSL 或普通虚拟机作为正式 ManiSkill 平台，因为 GUI、Vulkan、GPU 仿真容易受限。

### 0.1 如果现在先用虚拟机调试

如果你暂时不想装双系统，可以先用 VMware Workstation / VirtualBox 安装 Ubuntu 24.04 LTS 虚拟机，用来做代码结构开发、Python 包安装、LLM 调用、日志系统、prompt/debug loop 等前期工作。

虚拟机适合：

- 写和调试 `maniskill_backend/` 的普通 Python 代码；
- 测试 `CapabilityCard`、`FailureReport`、LMP executor、LLM client；
- 跑不依赖 GPU 渲染的单元测试；
- 整理实验日志、prompt、baseline 流程；
- 在 Ubuntu 环境里验证依赖安装流程。

虚拟机不适合：

- 作为正式 ManiSkill GPU simulation 平台；
- 依赖 NVIDIA GPU / Vulkan 的 GUI 渲染；
- 跑最终实验矩阵；
- 判断 viewer、GPU sim、SAPIEN 渲染是否真的稳定。

虚拟机建议配置：

- Ubuntu Desktop 24.04 LTS；
- CPU: 4 cores 或更多；
- RAM: 8-16 GB；
- Disk: 80-120 GB；
- 开启 3D acceleration；
- 安装 VMware Tools / VirtualBox Guest Additions；
- 项目放在虚拟机 Linux 文件系统内，例如 `~/Embodied/embodied_migration`，不要长期直接在共享文件夹里跑 pip/conda 环境。

虚拟机里可以运行同一个脚本，但建议先跳过 NVIDIA driver：

```bash
bash scripts/setup_ubuntu_maniskill.sh --yes --no-driver
```

如果后续 `mani_skill` 的纯导入、代码结构和非渲染测试都没问题，再把同一套仓库切到原生 Ubuntu 或外置 SSD Ubuntu 上跑正式 GPU/GUI 测试。

准备材料：

- 16 GB 或更大的 U 盘一个，制作启动盘会清空 U 盘；
- 至少 100 GB 空闲磁盘空间，建议 150-200 GB；
- Windows 重要文件备份；
- 笔记本接电源；
- 如果 Windows 开了 BitLocker，先保存恢复密钥。

安装前在 Windows 中做两件事：

1. 打开“磁盘管理”，从空间充足的分区压缩出一块未分配空间，建议 150 GB 左右。不要在 Windows 里把这块空间格式化成新盘。
2. 关闭 Windows 快速启动，避免双系统访问磁盘时状态不一致。

安装流程：

1. 从 Ubuntu 官方下载 Ubuntu Desktop 24.04 LTS ISO。
2. 用官方推荐的启动盘工具或 Rufus / balenaEtcher 把 ISO 写入 U 盘。
3. 重启电脑，进入启动菜单，选择 U 盘启动。
4. 进入 Ubuntu installer 后，先选择 “Try Ubuntu” 测试键盘、触摸板、Wi-Fi、屏幕是否正常。
5. 确认正常后点击 “Install Ubuntu”。
6. 安装类型优先选择 “Install Ubuntu alongside Windows Boot Manager”。如果没有这个选项，再手动选择刚才压缩出的未分配空间。
7. 安装完成后拔掉 U 盘并重启，启动菜单中应能选择 Ubuntu 或 Windows。

装完系统后，先在 Ubuntu 里运行：

```bash
sudo apt update
sudo apt upgrade -y
sudo reboot
```

然后再回到下面的项目环境安装步骤。

## 1. 在 Ubuntu 中克隆项目

建议把项目放在 Ubuntu 自己的文件系统中，例如 `~/Embodied`，不要直接在 NTFS 分区或 WSL 挂载目录里跑正式仿真。

```bash
mkdir -p ~/Embodied
cd ~/Embodied
git clone https://github.com/hexizou-730/embodied_migration.git
cd embodied_migration
```

如果你已经用 GitHub Desktop 或 Windows 拉过一次，也仍建议在 Ubuntu 里再克隆一份正式实验副本。

## 2. 一键安装

```bash
cd ~/Embodied/embodied_migration
bash scripts/setup_ubuntu_maniskill.sh --yes
```

脚本默认会尝试安装推荐的 NVIDIA 桌面驱动。安装驱动后通常需要重启：

```bash
sudo reboot
```

重启后回到项目目录，检查环境：

```bash
cd ~/Embodied/embodied_migration
conda activate em-ms
python -c "import mani_skill, torch; print('mani_skill ok'); print('cuda:', torch.cuda.is_available())"
nvidia-smi
vulkaninfo --summary
```

## 3. 如果想分步执行

只安装 conda 和 Python 包，跳过 NVIDIA driver：

```bash
bash scripts/setup_ubuntu_maniskill.sh --yes --no-driver
```

先升级系统包再安装：

```bash
bash scripts/setup_ubuntu_maniskill.sh --yes --upgrade
```

如果 `--upgrade` 后提示需要重启，先重启，再重新运行脚本。

使用别的 conda 路径：

```bash
bash scripts/setup_ubuntu_maniskill.sh --yes --conda-dir "$HOME/miniforge3"
```

## 4. 第一条 ManiSkill 验证命令

这是下一阶段的 smoke test：

```bash
conda activate em-ms
python -m mani_skill.examples.demo_random_action -e PickCube-v1 --render-mode human
```

期望现象：SAPIEN/ManiSkill viewer 打开，`PickCube-v1` 场景开始随机动作。

## 5. 常见问题

### `nvidia-smi` 找不到或失败

如果刚安装完驱动但还没重启，这是正常的。先执行：

```bash
sudo reboot
```

重启后再运行：

```bash
nvidia-smi
```

### Secure Boot 相关问题

Ubuntu 官方的 `ubuntu-drivers` 默认更适合 Secure Boot 场景，因为它优先使用 Canonical 打包和签名的驱动模块。如果重启后驱动仍不可用，检查 BIOS/UEFI Secure Boot 状态，以及安装过程中是否出现 MOK enrollment 提示。

### `vulkaninfo` 失败

ManiSkill 渲染需要 Vulkan。先确认驱动正常：

```bash
nvidia-smi
```

然后确认 Vulkan ICD 文件是否存在：

```bash
ls /usr/share/vulkan/icd.d/
ls /usr/share/glvnd/egl_vendor.d/
```

如果没有 NVIDIA 相关文件，通常是驱动安装或重启没有完成。

### WSL 提示退出

脚本会主动拒绝在 WSL 中继续，因为项目计划书明确建议正式实验使用原生 Ubuntu。WSL 可以用于编辑代码，但不作为 GUI/GPU 仿真的主环境。
