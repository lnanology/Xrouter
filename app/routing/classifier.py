"""Backward-compatible re-export. The canonical implementation moved to
`app.intelligence.complexity` as part of Phase 2 (section 三十六's "Task
Classifier" / "Complexity Engine"), which also adds task-type detection on
top of this same complexity score. Kept here so existing Phase 1 call sites
and tests keep working unchanged."""
from __future__ import annotations

from app.intelligence.complexity import HARD_KEYWORDS, classify_complexity

__all__ = ["classify_complexity", "HARD_KEYWORDS"]
