from __future__ import annotations

import unittest

from maniskill_backend.task_catalog import load_task_catalog


class TaskCatalogTests(unittest.TestCase):
    def test_catalog_has_twenty_target_independent_tasks(self) -> None:
        catalog = load_task_catalog()
        self.assertEqual(catalog["task_count"], 20)
        self.assertEqual(len(catalog["tasks"]), 20)
        self.assertEqual(catalog["target_robot_policy"], "runtime_input_not_catalogued")
        for task in catalog["tasks"]:
            self.assertNotIn("target_robot", task)
            self.assertNotIn("case_id", task)

    def test_every_task_has_six_ordered_variants(self) -> None:
        catalog = load_task_catalog()
        for task in catalog["tasks"]:
            variants = task["difficulty_variants"]
            self.assertEqual([item["difficulty"] for item in variants], list(range(1, 7)))
            limits = [item["episode_limit"] for item in variants]
            self.assertEqual(limits, sorted(limits, reverse=True))
            self.assertEqual(limits[-1], task["episode_limit"])

    def test_success_always_comes_from_official_evaluate(self) -> None:
        catalog = load_task_catalog()
        for task in catalog["tasks"]:
            self.assertEqual(task["success_signal"]["authority"], "env.unwrapped.evaluate()")
            self.assertEqual(task["success_signal"]["key"], "success")


if __name__ == "__main__":
    unittest.main()
