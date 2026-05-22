# Mac 迁移项目指导

这份文档用于把当前项目从 Windows/WSL 工作习惯迁移到 Mac。当前实验
不是在 Mac 本机跑，而是从 Mac 连接 Polytechnique 远程 GPU 机器运行。

## 1. 当前结论

可以把 Mac 当作主力开发电脑：

- Mac 负责 SSH、VS Code Remote SSH、看日志、改代码、同步 GitHub。
- 远程 Linux GPU 机器负责 ManiSkill 仿真和 LLM 迭代实验。
- GitHub 用于同步本地 Mac、远程机器和其他电脑上的代码。

当前不要把 Mac 本机结果当作仿真实验结果。实验结果以远程 GPU 机器
运行输出为准。

## 2. 当前远程机器

当前可用并已经配置过的实验机器是：

```text
SSH host: rotule.polytechnique.fr
SSH user: hexi.zou
remote project: ~/Embodied/embodied_migration
conda env: em-ms
```

在 `rotule` 上已经看到：

```text
OS: AlmaLinux 9.7
GPU: NVIDIA RTX 4000 Ada Generation
NVIDIA driver: 595.71.05
CUDA reported by nvidia-smi: 13.2
Vulkan ICD: /etc/vulkan/icd.d/nvidia_icd.json
```

之前也试过 `allemagne.polytechnique.fr`，但当时 GPU 被别的进程占用，
并且 Vulkan ICD 配置不如 `rotule` 直接可用。当前项目默认继续用
`rotule`。

## 3. Mac 首次连接

先在 Mac Terminal 测试 SSH：

```bash
ssh hexi.zou@rotule.polytechnique.fr
```

第一次连接如果出现 host authenticity 提示，确认主机名是
`rotule.polytechnique.fr` 后输入：

```text
yes
```

然后输入学校账号密码。

为了以后命令更短，可以在 Mac 的 `~/.ssh/config` 中加入：

```sshconfig
Host poly-rotule
    HostName rotule.polytechnique.fr
    User hexi.zou
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

以后连接：

```bash
ssh poly-rotule
```

## 4. Mac 上用 VS Code

1. 在 Mac 安装 VS Code。
2. 安装 VS Code 扩展 `Remote - SSH`。
3. 打开 Command Palette。
4. 选择 `Remote-SSH: Connect to Host...`。
5. 选择 `poly-rotule`，或直接输入：

```text
hexi.zou@rotule.polytechnique.fr
```

6. 连接后打开远程目录：

```text
~/Embodied/embodied_migration
```

在 VS Code 左下角看到远程 SSH 标识后，终端、Python、Git 和文件编辑
都发生在远程机器上，不是在 Mac 本地。

## 5. 每次开始工作

登录 `rotule` 后执行：

```bash
cd ~/Embodied/embodied_migration
git pull
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate em-ms
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

检查环境：

```bash
nvidia-smi
python --version
python -m unittest discover -s tests -v
```

如果 xarm6 planner 报 NumPy 2.x 与 `mplib`/`toppra` 不兼容，执行：

```bash
pip install "numpy>=1.24,<2" --force-reinstall
```

## 6. LLM 配置

远程项目目录需要 `.env`。不要把 `.env` 提交到 GitHub。

最少需要：

```dotenv
OPENROUTER_API_KEY=your_key_here
EM_MODEL=anthropic/claude-sonnet-4.5
```

模型可以临时覆盖，例如：

```bash
export EM_MODEL=anthropic/claude-opus-4.7
```

如果 OpenRouter 模型名调整，以 OpenRouter 当时可用的模型 ID 为准。

## 7. 当前研究目标

当前研究目标已经收敛为：

```text
full-stack cross-embodiment robot migration
```

实验问题：

```text
一个 Panda 程序已经能在仿真中完成任务。
迁移到 xarm6 时，哪些内容必须一起迁移，
才能让目标机器人在真实 ManiSkill 仿真中完成相同任务？
```

当前主流程：

1. Panda 成功源代码。
2. Panda 源 skill wrapper 在 ManiSkill 中成功执行。
3. xarm6 跑 source-copy，暴露程序层和执行层差异。
4. LLM 根据真实失败日志判断下一轮该改目标 LMP、skill wrapper、profile
   还是 controller route。
5. 项目自动应用受约束的 LLM patch，先跑测试，再跑 xarm6 仿真。
6. 直到成功或修复预算耗尽。
7. 同时记录 LMP 代码改动和 skill-wrapper / controller 改动。

Capability Card 和 Failure Report 现在是辅助上下文，不再是主要研究
对象。

## 8. 代码层次

当前 LLM runner 写的是高层 LMP 代码，例如：

```python
ok = robot.hook_object(tool, cube)
if ok:
    ret_val = robot.pull_with_tool(tool, cube, workspace)
else:
    ret_val = "failure: hook"
```

这些高层 API 由项目里的 ManiSkill skill wrapper 转成真实仿真动作：

```text
LLM target code
  -> high-level skill API
  -> maniskill_backend/skill_adapter.py
  -> ManiSkill planner / env.step(action)
  -> real simulator feedback
```

因此实验里要区分两类迁移对象和失败：

- 高层程序迁移：顺序、参数、调用 API 和任务分支。
- 执行层迁移：目标机器人需要不同的抓取、规划、接触轨迹、TCP 补偿或
  控制 primitive，高层代码本身不足以修复。

这一区分不是把执行层排除掉，而是让 LLM 在失败反馈下选择要迁移哪一层。
只迁移高层代码不足以说明跨 embodiment 迁移已经完成。

## 9. 当前任务与机器人

当前主要机器人：

| Robot | Role |
|---|---|
| `panda` | 成功源机器人 |
| `xarm6_robotiq` | 目标迁移机器人 |

当前任务：

| Project task | ManiSkill env | Current use |
|---|---|---|
| `pick_cube` | `PickCube-v1` | 基础迁移和 controller/path smoke test |
| `stack_cube` | `StackCube-v1` | 支撑任务，用于检查 stacking 路径 |
| `pull_cube_tool` | `PullCubeTool-v1` | **Case 01** 完整 Panda 到 xarm6 迁移 |
| `peg_insertion` | `PegInsertionSide-v1` | 暂停，官方 Panda solver 在 seed 0 也失败 |

注意：

- 项目命令使用清晰 task id，如 `pull_cube_tool`。
- ManiSkill 官方环境名仍然带 `-v1`，如 `PullCubeTool-v1`。
- `PullCubeTool-v1` 官方支持机器人主要是 Panda/Fetch。xarm6 运行时出现
  unsupported robot warning 是当前实验中的已知现象。

## 10. 当前实验事实

已经确认：

```text
pick_cube + panda + pd_ee_delta_pos -> success
pick_cube + xarm6_robotiq + pd_joint_pos planner -> success
stack_cube + panda + pd_joint_pos -> success
stack_cube + xarm6_robotiq + pd_joint_pos -> success
pull_cube_tool + official Panda solver -> success at seed 0
peg_insertion + official Panda solver -> failure at seed 0, parked
```

`pull_cube_tool` 的 xarm6 仍在推进中。当前已经观察到：

1. Panda source 成功。
2. xarm6 高层 source-copy 能走到工具使用 wrapper。
3. xarm6 多次失败不是简单的 pull distance 不够。
4. 日志曾显示 `cube_delta=[0,0,0]`，说明方块没有被工具带动。
5. 日志也显示工具抓取检查可通过，但工具位置修正时工具未稳定跟随
   TCP，说明 xarm6 的工具抓取 primitive / skill wrapper 需要迁移。

因此 `pull_cube_tool` 当前更像：

```text
Panda task stack
  -> xarm6 program migration
  -> xarm6 tool-wrapper / grasp / contact primitive migration
  -> full target simulation success
```

它已经固定为第一个正式案例：

```text
case id: case01_pull_cube_tool_panda_to_xarm6
source robot: panda
target robot: xarm6_robotiq
source/target control: pd_joint_pos
first seed: 0
episode budget: 300
```

这个案例必须同时给出两类证据：LLM 目标 LMP 代码差异，以及
LLM 对 `skill_adapter.py` / controller primitive 的目标执行层迁移差异。

## 11. 重要运行命令

Case 01 自动跨层迁移主入口：

```bash
python -m maniskill_backend.full_stack_runner \
  --case case01_pull_cube_tool_panda_to_xarm6 \
  --max-repair-rounds 3 \
  --sim-backend auto \
  --render-backend gpu
```

它要求 Git tracked worktree 是干净的；成功 patch 会留在本地 diff 里供
检查和提交，测试失败的 patch 会自动回退。

最小 Panda 源端检查：

```bash
python -m maniskill_backend.real_runner \
  --task pull_cube_tool \
  --robot panda \
  --method source-copy \
  --seed 0 \
  --control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 300
```

xarm6 单次目标执行：

```bash
python -m maniskill_backend.real_runner \
  --task pull_cube_tool \
  --robot xarm6_robotiq \
  --method source-copy \
  --seed 0 \
  --control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 300
```

当前主实验 runner：

```bash
python -m maniskill_backend.iterative_runner \
  --task pull_cube_tool \
  --source-robot panda \
  --target-robot xarm6_robotiq \
  --max-attempts 3 \
  --seed 0 \
  --target-control-mode pd_joint_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 300
```

## 12. 结果文件

真实仿真 trial：

```text
results/real_trials.jsonl
results/real_trials.md
results/real_summary.csv
```

迭代 LLM 实验：

```text
results/iterative_trials.jsonl
results/iterative_trials.md
results/iterative_summary.csv
```

推荐阅读顺序：

1. `results/iterative_summary.csv` 看汇总。
2. `results/iterative_trials.md` 看每次 LLM 代码和日志。
3. `results/iterative_trials.jsonl` 做后续统计脚本。

## 13. 迁移到 Mac 后的下一步

建议顺序：

1. 用 Mac SSH / VS Code Remote SSH 成功连接 `rotule`。
2. 在远程项目目录 `git pull`。
3. 跑单元测试。
4. 跑 `pick_cube` 或 `stack_cube` 复现已知成功结果。
5. 继续调 `pull_cube_tool` 的 xarm6 工具抓取和接触 wrapper。
6. wrapper 稳定后，再跑 iterative LLM 的多 seed / 多 attempt 统计。

当前不建议立刻做大量重复实验，因为 `pull_cube_tool` 的 xarm6 技能层
还在迁移中。先保证任务执行通路可信，再扩大实验数量。
