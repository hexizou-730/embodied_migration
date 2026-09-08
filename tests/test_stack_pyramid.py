"""Interface/state-machine tests, not a physics or task-success benchmark."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lmp.executor import execute_lmp
from maniskill_backend.cases import find_full_migration_case
from maniskill_backend.failure_diagnosis import diagnose_failure
from maniskill_backend.module_generation_runner import (
    build_module_generation_prompt, guarded_adapter_validation_error,
    run_module_generation_migration, validate_generated_adapter_module,
)
from maniskill_backend.paper_benchmark import build_paper_plan
from maniskill_backend.real_runner import run_real_code_trial
from maniskill_backend.skill_adapter import ManiSkillSceneAdapter
from maniskill_backend.stack_pyramid import ManiSkillStackPyramidRobot, stack_observation
from maniskill_backend.tasks import get_task_spec
from scripts.agent_migration_runner import _module_generation_command, _has_structured_probe, build_arg_parser


ROOT = Path(__file__).resolve().parents[1]
CASE = find_full_migration_case("stack", "panda", "fetch")


class FakeCube:
    def __init__(self, xyz):
        self.pose = SimpleNamespace(p=np.array([xyz], dtype=np.float32), q=np.array([[1, 0, 0, 0]]))
        self.static = True

    def is_static(self, **kwargs):
        return np.array([self.static])


class FakeEnv:
    """Kinematic test double. No collision/force or GPU success evidence."""
    def __init__(self, dim=4, limit=800, grasp=True):
        self.action_space = SimpleNamespace(shape=(dim,), dtype=np.float32,
                                            low=np.full(dim, -1.), high=np.full(dim, 1.))
        self.cubeA = FakeCube([0.1, -0.03, 0.02])
        self.cubeB = FakeCube([-0.05, 0.04, 0.02])
        self.cubeC = FakeCube([0.05, 0.15, 0.02])
        self.cube_half_size = np.array([0.02] * 3)
        self.agent = SimpleNamespace(tcp=FakeCube([0., 0., 0.22]),
                                     is_grasping=lambda actor: np.array([actor is self.held]))
        self.held, self.grasp_enabled = None, grasp
        self.count, self.limit = 0, limit
        self.unwrapped = self

    def evaluate(self):
        a, b, c = (v.pose.p[0] for v in (self.cubeA, self.cubeB, self.cubeC))
        ok = (np.linalg.norm(a[:2] - b[:2]) < 0.062 and
              np.linalg.norm(c[:2] - (a[:2] + b[:2]) / 2) < 0.015 and
              abs(c[2] - 0.06) < 0.012 and self.held is None and self.cubeC.static)
        return {"success": np.array([ok])}

    def step(self, action):
        if self.count >= self.limit:
            raise AssertionError("env.step called after termination")
        self.count += 1
        delta = np.asarray(action).reshape(-1)[:3] * 0.02
        self.agent.tcp.pose.p += delta
        if self.held is not None:
            self.held.pose.p += delta
        if action[3] > 0:
            self.held = None
        elif self.held is None and self.grasp_enabled:
            for actor in (self.cubeA, self.cubeB, self.cubeC):
                if np.linalg.norm(actor.pose.p - self.agent.tcp.pose.p) < 0.012:
                    self.held = actor
                    break
        return None, 0., False, self.count >= self.limit, {
            **self.evaluate(), "elapsed_steps": np.array([self.count]),
        }

    def reset(self, seed=None):
        return {}, {"elapsed_steps": np.array([0]), **self.evaluate()}


def robot(env):
    return ManiSkillStackPyramidRobot(env, robot_uid="panda" if env.action_space.shape[-1] == 4 else "fetch",
                                     control_mode="pd_ee_delta_pos")


class StackPyramidTest(unittest.TestCase):
    def test_registry_aliases_and_frozen_program(self):
        task = get_task_spec("StackPyramid-v1")
        self.assertEqual(task.task_id, CASE.task_id)
        self.assertEqual(task.target_robots, ("panda", "fetch"))
        self.assertEqual(CASE.max_episode_steps, 800)
        self.assertIn(task.source_program.strip(), (ROOT / CASE.target_program_path).read_text())
        validate_generated_adapter_module((ROOT / CASE.seed_adapter_path).read_text())
        self.assertFalse(_has_structured_probe(CASE))

    def test_paper_default_matrix_stays_frozen_and_exploration_is_blocked(self):
        plan = build_paper_plan(tier_id="pilot")
        self.assertEqual(len(plan["case_ids"]), 5)
        self.assertNotIn(CASE.case_id, plan["case_ids"])
        plan = build_paper_plan(tier_id="pilot", case_ids=[CASE.case_id])
        self.assertTrue(all(not r["ready"] for r in plan["runs"]))

    def test_action_mapping_and_batch_rejection(self):
        for dim in (4, 9):
            env = FakeEnv(dim)
            r = robot(env)
            action = r._make_action(np.array([2., 0., -2.]), gripper=-1)
            np.testing.assert_equal(action[:4], [1, 0, -1, -1])
            self.assertTrue(np.all(action[4:] == 0))
        env.action_space.shape = (2, 9)
        with self.assertRaises(ValueError):
            robot(env)

    def test_sequence_uses_multi_object_state_and_records_release(self):
        env = FakeEnv()
        r = robot(env)
        ok, msg, loc = execute_lmp(get_task_spec("stack").source_program,
                                   {"robot": r, "scene": ManiSkillSceneAdapter()}, verbose=False)
        self.assertTrue(ok, msg)
        self.assertTrue(loc["ret_val"], r.execution_log())
        stages = [s["stage"] for s in r.stage_trace]
        self.assertLess(stages.index("base.verify"), stages.index("top.approach"))
        self.assertIn("top.release", stages)
        self.assertIn("top.settle", stages)
        self.assertIsNone(env.held)
        self.assertEqual(set(stack_observation(r)["cubes"]), {"cubeA", "cubeB", "cubeC"})
        json.dumps(r.stage_trace)

    def test_failed_base_prevents_top_stage(self):
        r = robot(FakeEnv(grasp=False))
        _, _, loc = execute_lmp(get_task_spec("stack").source_program,
                               {"robot": r, "scene": ManiSkillSceneAdapter()}, verbose=False)
        self.assertFalse(loc["ret_val"])
        self.assertEqual(r.stage, "base.close")
        self.assertFalse(any(s["stage"].startswith("top.") for s in r.stage_trace))

    def test_budget_termination_never_steps_again(self):
        env = FakeEnv(limit=3)
        r = robot(env)
        execute_lmp(get_task_spec("stack").source_program,
                    {"robot": r, "scene": ManiSkillSceneAdapter()}, verbose=False)
        self.assertEqual(env.count, 3)
        self.assertTrue(r._early_stop())
        state = stack_observation(r)
        diagnosis = diagnose_failure(task_id="stack_pyramid", success=False, runtime_diagnostics=state)
        self.assertEqual(diagnosis["reason"], "stack_episode_budget_exhausted")

    def test_base_rechecked_before_top_and_static_gate(self):
        env = FakeEnv()
        r = robot(env)
        scene = ManiSkillSceneAdapter()
        a, b, c = (scene.get_object(n) for n in ("cubeA", "cubeB", "cubeC"))
        self.assertTrue(r.prepare_base(a, b))
        env.cubeB.pose.p[0, 0] += 0.15
        self.assertFalse(r.stack_on(c, a, b))
        self.assertEqual(r.stage, "base.verify")
        env.cubeB.pose.p[0, 0] -= 0.15
        env.cubeA.static = False
        self.assertFalse(stack_observation(r)["base_ready"])

    def test_prompt_uses_stack_context_and_forwards_stage_feedback(self):
        result = {"runtime_diagnostics": {"stage": "top.release"}, "stage_trace": [{"stage": "base.verify"}]}
        for attempts in ([], [{"round": 1}]):
            prompt = build_module_generation_prompt(case=CASE, target_result=result, attempts=attempts)
            self.assertIn("cubeC", prompt)
            self.assertIn("top.release", prompt)
            self.assertIn("prepare_base", prompt)
            self.assertIn("stack_on", prompt)
            self.assertNotIn("farther positive-x", prompt)
        self.assertIn("override", guarded_adapter_validation_error((ROOT / CASE.seed_adapter_path).read_text(), CASE))

    def test_source_failure_stops_generation_and_budget_is_applied(self):
        with patch("maniskill_backend.module_generation_runner._run_source_trial", return_value={"success": False}) as source, \
             patch("maniskill_backend.module_generation_runner.gen_text") as llm, \
             patch("maniskill_backend.module_generation_runner._git_diff", return_value=""):
            result = run_module_generation_migration(case_id=CASE.case_id, seed=7, max_episode_steps=900)
        self.assertFalse(result["success"])
        self.assertEqual(source.call_args.kwargs["case"].max_episode_steps, 900)
        self.assertEqual(source.call_args.kwargs["seed"], 7)
        llm.assert_not_called()

    def test_agent_passes_seed_budget_and_forces_real_generation(self):
        args = build_arg_parser().parse_args(["--seed", "7", "--max-episode-steps", "900"])
        command = _module_generation_command(args, CASE, Path("cycle"))
        self.assertEqual(command[command.index("--seed") + 1], "7")
        self.assertEqual(command[command.index("--max-episode-steps") + 1], "900")
        self.assertIn("--force-regeneration", command)
        self.assertIn("--no-analysis", command)

    def test_real_runner_rejects_true_return_without_official_success(self):
        env = FakeEnv(9)
        r = robot(env)
        r.prepare_base = lambda *a: True
        r.stack_on = lambda *a: True
        with patch("maniskill_backend.real_runner.ManiSkillEnvAdapter") as adapter, \
             patch("maniskill_backend.real_runner._build_robot_adapter_from_module", return_value=r), \
             patch("maniskill_backend.real_runner.observe_runtime_contract", return_value={}):
            adapter.return_value.make.return_value = env
            adapter.return_value.reset.return_value = env.reset()
            result = run_real_code_trial(task_id=CASE.task_id, robot_uid="fetch", method="test",
                                         code=get_task_spec("stack").source_program, adapter_module="mock")
        self.assertFalse(result["success"], result)
        self.assertIn("official StackPyramid", result["message"])
        self.assertEqual(adapter.call_args.kwargs["reward_mode"], "sparse")
        self.assertEqual(set(result["runtime_diagnostics"]["cubes"]), {"cubeA", "cubeB", "cubeC"})

    def test_agent_cli_dry_run_and_unsupported_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "migrate.py", "--task", "stack", "--target", "fetch",
                       "--mode", "agent", "--dry-run", "--output-root", directory, "--run-name", "preview"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            log = (Path(directory) / "preview/agent_loop/commands.log").read_text()
            self.assertIn("--max-episode-steps 800", log)
            self.assertIn("--force-regeneration", log)
            command[command.index("--mode") + 1] = "online"
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("not implemented", result.stderr)


if __name__ == "__main__":
    unittest.main()
