"""
Code extraction: 从 LLM 返回的文本中提取可执行的 Python 代码。

LLM 有时会用 ```python ... ``` 包裹,有时会直接输出裸代码。
"""
import re
from typing import Optional


_FENCED = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> Optional[str]:
    """优先提取三反引号围栏内的代码;否则返回 None (让调用方判断是否用裸文本)。"""
    m = _FENCED.search(text)
    if m:
        return m.group(1).strip()
    return None


def extract_code_or_text(text: str) -> str:
    """提取代码;如果没围栏就假设整段都是代码。"""
    code = extract_code(text)
    return code if code is not None else text.strip()
