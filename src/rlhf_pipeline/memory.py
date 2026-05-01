from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_env import SyntheticTask


LANGUAGE_PREFERENCES: tuple[str, ...] = ("Python", "Rust", "TypeScript")
STYLE_PREFERENCES: tuple[tuple[str, str], ...] = (
    ("concise", "User prefers concise answers."),
    ("step_by_step", "User prefers step-by-step explanations."),
)


def synthetic_memory_profile(variant: int) -> dict[str, str]:
    language = LANGUAGE_PREFERENCES[variant % len(LANGUAGE_PREFERENCES)]
    style_key, style_text = STYLE_PREFERENCES[variant % len(STYLE_PREFERENCES)]
    return {
        "answer_style": style_text,
        "skill_level": "User is beginner-level.",
        "preferred_language": f"User prefers {language} examples.",
        "comparison_focus": "User wants cost-focused comparisons.",
        "safety_preference": "User prefers safety-first answers.",
        "_style_key": style_key,
        "_language": language,
    }


def relevant_memory_entries(task: "SyntheticTask") -> dict[str, str]:
    keys = task.requirements.get("memory_keys", ())
    return {
        str(key): value
        for key in keys
        if (value := task.memory.get(str(key))) is not None
    }


def memory_lookup_output(task: "SyntheticTask") -> str:
    entries = relevant_memory_entries(task)
    if not entries:
        return "No relevant user memory was found."
    return "\n".join(f"{key}: {value}" for key, value in entries.items())

