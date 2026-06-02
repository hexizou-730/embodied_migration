# 运行指南：PullCube 迁移实验

当前项目只保留一条主线：

```text
任务：PullCube-v1 / pull_cube
源机器人：panda
目标机器人：fetch
目标：把 Panda 上的拉方块任务迁移到 Fetch
```

## 1. 进入项目和环境

```bash
cd ~/Embodied/embodied_migration
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate em-ms
```

如果在 Linux/NVIDIA 服务器上运行：

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

预期现象：

```text
(em-ms) ... embodied_migration $
```

## 2. 检查依赖

```bash
python -m maniskill_backend.sim_check \
  --env PullCube-v1 \
  --robot panda \
  --obs-mode state \
  --control-mode pd_ee_delta_pos
```

预期输出包含：

```text
maniskill import ok
env reset ok
```

如果这里报 ManiSkill / Vulkan / numpy 错误，先修环境，不要继续跑迁移实验。

## 3. 跑单元测试

```bash
python -m unittest discover -s tests -v
```

预期输出最后是：

```text
OK
```

这些测试会确认：

- 当前只接受 `pull_cube`；
- 当前使用 `panda`、`fetch` 和 `xarm6_robotiq`；
- 旧的 `pull_cube_tool`、`stack_cube`、`peg_insertion` 等任务已不在主线；
- 生成 adapter 的接口仍然可用。

## 4. 配置 DeepSeek API

如果你要让 LLM 参与生成 xarm6 adapter，先创建 `.env`：

```bash
cp .env.example .env
```

然后编辑 `.env`，填入：

```text
EM_LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的_deepseek_key
EM_MODEL=deepseek-v4-pro
EM_MAX_TOKENS=8192
EM_DEEPSEEK_THINKING=disabled
```

检查 key 是否被项目识别：

```bash
python - <<'PY'
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path.cwd() / ".env")
from maniskill_backend.llm import has_llm_key
from llm_client import current_provider, default_model
print("provider =", current_provider())
print("model =", default_model())
print("has_llm_key =", has_llm_key())
from llm_client import completion_token_limit, deepseek_thinking_mode
print("max_tokens =", completion_token_limit())
print("thinking =", deepseek_thinking_mode())
PY
```

预期输出：

```text
provider = deepseek
model = deepseek-v4-pro
has_llm_key = True
max_tokens = 8192
thinking = disabled
```

## 5. 跑 Panda 源端

```bash
python -m maniskill_backend.real_runner \
  --task pull_cube \
  --robot panda \
  --method source-copy \
  --seed 0 \
  --control-mode pd_ee_delta_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 100
```

预期输出会打印 JSON 结果，关键字段类似：

```text
"task_id": "pull_cube"
"robot_uid": "panda"
"method": "source-copy"
"success": true
```

如果 Panda 源端失败，说明源任务还没有建立好，不能进入迁移对比。

## 6. 跑 xarm6 目标端 source-copy

```bash
python -m maniskill_backend.real_runner \
  --task pull_cube \
  --robot xarm6_robotiq \
  --method source-copy \
  --seed 0 \
  --control-mode pd_ee_delta_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 500
```

可能输出两种结果。

成功时：

```text
"robot_uid": "xarm6_robotiq"
"success": true
```

失败时：

```text
"success": false
"failure_type": "contact execution failure" 或其他失败类型
"failure_layer": "skill_adapter" / "controller_primitive" / "task_outcome"
```

这一步的作用是得到目标机器人真实失败日志。

## 7. 跑主实验：LLM 生成 xarm6 adapter

```bash
python -m maniskill_backend.module_generation_runner \
  --case case02_pull_cube_panda_to_xarm6 \
  --max-attempts 3 \
  --sim-backend auto \
  --render-backend gpu
```

预期流程：

```text
1. 检查 Panda 源端
2. 检查 xarm6 目标端
3. 如果 xarm6 失败，把 failure log 发给 LLM
4. LLM 生成完整 Python adapter module
5. 写入 maniskill_backend/generated_adapters/case02_xarm6_pull_cube.py
6. 跑单元测试
7. 用生成的 adapter 重新跑 xarm6
8. 写入结果文件
```

输出文件：

```text
results/module_generation_trials.jsonl
results/module_generation_trials.md
```

## 8. 单独测试当前生成的 xarm6 adapter

```bash
python -m maniskill_backend.real_runner \
  --task pull_cube \
  --robot xarm6_robotiq \
  --method target-module-generation \
  --seed 0 \
  --control-mode pd_ee_delta_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 500 \
  --code-file maniskill_backend/case_programs/case01_pull_cube.py \
  --adapter-module maniskill_backend.generated_adapters.case02_xarm6_pull_cube
```

预期输出：

```text
"method": "target-module-generation"
"robot_uid": "xarm6_robotiq"
"success": true/false
```

如果失败，重点看：

```text
"failure_type"
"failure_layer"
"message"
"execution_log"
"final_info"
```

这些字段就是后续写论文时的失败分析证据。

## 9. 跑 program-only baseline

```bash
python -m maniskill_backend.iterative_runner \
  --task pull_cube \
  --source-robot panda \
  --target-robot xarm6_robotiq \
  --max-attempts 3 \
  --seed 0 \
  --target-control-mode pd_ee_delta_pos \
  --sim-backend auto \
  --render-backend gpu \
  --max-episode-steps 500
```

这个 baseline 只让 LLM 改高层 LMP 代码，不改 skill adapter。

如果它失败而 module generation 成功，就能说明：

```text
program-only migration 不够；
目标机器人还需要 skill/contact/controller 层迁移。
```

## 10. 看结果

```bash
tail -n 120 results/module_generation_trials.md
git diff -- maniskill_backend/generated_adapters/case02_xarm6_pull_cube.py
git status --short
```

预期能看到：

- 每轮是否调用了 LLM；
- 每轮生成的 adapter 是否通过测试；
- xarm6 最终是否成功；
- 生成 adapter 和默认 adapter 的代码差异。

## 11. 推荐演示顺序

给老师演示时按这个顺序：

1. 打开 `README.md`，说明当前主实验是 `Panda -> xarm6_robotiq` 的 `PullCube-v1`。
2. 展示源程序只有 `robot.pull(cube, goal)`。
3. 跑 `python -m unittest discover -s tests -v`。
4. 跑 Panda source-copy，证明源端可行。
5. 跑 xarm6 source-copy，展示目标端成功或失败日志。
6. 跑 `module_generation_runner`，展示 LLM 生成目标 adapter。
7. 打开 `results/module_generation_trials.md`，解释迁移差异。
