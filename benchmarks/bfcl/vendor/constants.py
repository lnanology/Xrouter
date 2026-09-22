"""Trimmed constants for the vendored BFCL AST checker -- everything
ast_checker.py and its type_convertor helpers actually import, copied
from bfcl_eval/constants/enums.py and bfcl_eval/constants/type_mappings.py
at the pinned commit (see ../vendor/NOTICE). The original model_config.py
(2240 lines, a per-model API registry) is deliberately NOT vendored --
see ast_checker.py's convert_func_name for why it isn't needed here."""
from __future__ import annotations

from enum import Enum


class Language(Enum):
    """Controls the type checking path the AST checker takes."""

    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"


JAVA_TYPE_CONVERSION = {
    "byte": int,
    "short": int,
    "integer": int,
    "float": float,
    "double": float,
    "long": int,
    "boolean": bool,
    "char": str,
    "Array": list,
    "ArrayList": list,
    "Set": set,
    "HashMap": dict,
    "Hashtable": dict,
    "Queue": list,
    "Stack": list,
    "String": str,
    "any": str,
}

JS_TYPE_CONVERSION = {
    "String": str,
    "integer": int,
    "float": float,
    "Bigint": int,
    "Boolean": bool,
    "dict": dict,
    "array": list,
    "any": str,
}
