# WSL2 前期开发环境

可以先用 WSL2 做前期开发。它比普通虚拟机轻，和 Windows/VS Code 配合更顺手，适合在正式原生 Ubuntu 准备好之前把项目框架搭起来。

WSL2 适合：

- Python 项目开发；
- `CapabilityCard`、`FailureReport`、LMP executor；
- prompt 拼接、代码抽取、LLM 调用；
- `llm_static_feedback` 静态 baseline；
- 日志系统、CSV/JSONL 结果保存；
- 单元测试和结果分析脚本；
- 可能的 headless/CPU 小规模 smoke test。

WSL2 不作为最终正式平台：

- 不把 WSL2 作为 ManiSkill GUI viewer 稳定性结论；
- 不把 WSL2 作为 Vulkan/SAPIEN 渲染稳定性结论；
- 不在 WSL2 里安装 Linux NVIDIA driver；
- 不用 WSL2 跑最终论文实验矩阵。

说明：Microsoft 和 NVIDIA 官方都支持 WSL2 中的 CUDA 开发工作流，但 ManiSkill/SAPIEN 的 GUI、Vulkan、viewer 稳定性仍建议最后在原生 Ubuntu 上验证。

## 1. 安装 Ubuntu 24.04 WSL2

在 Windows PowerShell 中进入仓库：

```powershell
cd F:\Embodied\embodied_migration
```

先检查：

```powershell
.\scripts\windows_prepare_wsl2.ps1
```

如果提示还没安装 Ubuntu，执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows_prepare_wsl2.ps1 -Install
```

也可以手动执行：

```powershell
wsl --install -d Ubuntu-24.04
```

如果 Windows 提示重启，先重启。然后打开 Ubuntu，创建 Linux 用户名和密码。

## 2. WSL2 内安装项目环境

进入 Ubuntu WSL2 终端后：

```bash
mkdir -p ~/Embodied
cd ~/Embodied
git clone https://github.com/hexizou-730/embodied_migration.git
cd embodied_migration
bash scripts/wsl2_bootstrap_dev.sh
```

这个脚本会调用：

```bash
bash scripts/setup_ubuntu_maniskill.sh --yes --no-driver --allow-wsl
```

它会跳过 NVIDIA driver 和原生 Ubuntu kernel headers，只安装前期开发依赖。

## 3. 验证

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

## 4. 项目放在哪里

推荐放在 WSL2 的 Linux 文件系统：

```text
~/Embodied/embodied_migration
```

不要长期在 `/mnt/f/...` 里跑 conda/pip 环境。Windows 盘挂载路径适合临时复制文件，但大量 Python 小文件、虚拟环境、缓存放在 Linux 文件系统里更快、更少权限问题。

## 5. 和 Windows/VS Code 配合

推荐用 VS Code 的 WSL 扩展打开：

```bash
cd ~/Embodied/embodied_migration
code .
```

这样编辑、终端、Python 解释器都在 WSL2 里，路径不会混乱。

## 6. 后续迁移到原生 Ubuntu

迁移原则：

- 代码用 Git：commit/push 后在原生 Ubuntu clone；
- 环境用脚本：不要复制 conda 环境，重新跑安装脚本；
- 数据用 `rsync` 或压缩包：复制 `results/`、日志、LLM 输出、实验表格。

