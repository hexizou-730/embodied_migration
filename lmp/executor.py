"""
LMP Executor: 安全执行 LLM 生成的 Python 代码片段。

参考 Code as Policies (Liang et al. 2023):
- 禁用 import / exec / eval / dunder 变量
- 注入受限的 globals (感知 API + 控制 API + numpy)
- 捕获异常,把 traceback 返回给 LLM 做修正
"""
import re
import traceback
import numpy as np
from typing import Any, Dict, Tuple


# 禁止的语法模式 (简单黑名单,生产环境应该用 AST 审查)
FORBIDDEN_PATTERNS = [
    r"\bimport\b",
    r"\b__\w+__\b",      # dunder
    r"\beval\b",
    r"\bexec\b",
    r"\bopen\b",
    r"\bcompile\b",
    r"\bsubprocess\b",
    r"\bos\.",
    r"\bsys\.",
]

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def is_safe(code: str) -> Tuple[bool, str]:
    """粗粒度安全检查。返回 (safe?, reason)。"""
    for pat in FORBIDDEN_PATTERNS:
        m = re.search(pat, code)
        if m:
            return False, f"Forbidden pattern matched: {m.group(0)!r}"
    return True, ""


def execute_lmp(
    code: str,
    globals_dict: Dict[str, Any],
    verbose: bool = True,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    执行 LLM 生成的代码。

    Returns:
        (success, message, locals_dict)
          success: 代码是否无异常执行完毕
          message: 成功时 = 'ok' 或 ret_val 的字符串形式; 失败时 = traceback
          locals_dict: 执行后 locals() (可从中取 ret_val)
    """
    safe, reason = is_safe(code)
    if not safe:
        return False, f"[SafetyError] {reason}", {}

    # 提供 numpy 给生成的代码 (不通过 import)
    g = dict(globals_dict)
    g.setdefault("np", np)
    g.setdefault("__builtins__", SAFE_BUILTINS)

    locals_dict: Dict[str, Any] = {}

    if verbose:
        print("─" * 60)
        print("🧠 LMP Code to execute:")
        print(code)
        print("─" * 60)

    try:
        exec(code, g, locals_dict)
    except Exception:
        tb = traceback.format_exc(limit=3)
        if verbose:
            print(f"❌ LMP execution failed:\n{tb}")
        return False, tb, locals_dict

    ret_val = locals_dict.get("ret_val", None)
    msg = "ok" if ret_val is None else f"ret_val={ret_val!r}"
    if verbose:
        print(f"✅ LMP executed successfully. {msg}")
    return True, msg, locals_dict
