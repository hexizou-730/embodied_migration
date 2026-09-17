"""Safety and structure checks for LLM-generated adapter modules."""

from __future__ import annotations

import ast
import re


_MODULE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ALLOWED_IMPORT_PREFIXES = (
    "__future__",
    "math",
    "typing",
    "numpy",
    "sapien",
    "mani_skill",
    "maniskill_backend.dynamic_adapter",
)
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "input", "open", "__import__"}
_FORBIDDEN_TEXT = (
    "subprocess",
    "os.",
    "sys.",
    "socket",
    "requests",
    "urllib",
    "shutil",
    "pathlib",
)
_FORBIDDEN_PATTERNS = tuple(
    (snippet, re.compile(rf"(?<![A-Za-z0-9_]){re.escape(snippet)}"))
    for snippet in _FORBIDDEN_TEXT
)


def extract_python_module(text: str) -> str:
    """Extract a complete Python module from an LLM response."""

    candidates = [match.group(1).strip() for match in _MODULE_FENCE.finditer(text)]
    if candidates:
        for candidate in candidates:
            if "def build_robot" in candidate:
                return candidate
        return candidates[0]
    return text.strip()


def validate_generated_adapter_module(code: str) -> None:
    """Reject unsafe or structurally invalid generated adapter modules."""

    if not code.strip():
        raise ValueError("Generated adapter module is empty.")
    for snippet, pattern in _FORBIDDEN_PATTERNS:
        if pattern.search(code):
            raise ValueError(f"Generated adapter module contains forbidden text: {snippet}")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Generated adapter module is not valid Python: {exc}") from exc

    has_factory = any(
        isinstance(node, ast.FunctionDef) and node.name == "build_robot"
        for node in tree.body
    )
    if not has_factory:
        raise ValueError(
            "Generated adapter module must define "
            "build_robot(env, *, control_mode, robot_uid)."
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _validate_import(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("Generated adapter module must use absolute imports only.")
            _validate_import(node.module or "")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALLS:
                raise ValueError(f"Generated adapter module calls forbidden function: {func.id}")


def _validate_import(module_name: str) -> None:
    if not module_name:
        raise ValueError("Generated adapter module contains an empty import.")
    if not any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in _ALLOWED_IMPORT_PREFIXES
    ):
        raise ValueError(f"Generated adapter module imports disallowed module: {module_name}")
