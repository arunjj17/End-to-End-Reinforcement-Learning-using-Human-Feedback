from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .agent_env import AgentTrajectory, SyntheticTask, synthetic_final_answer, visible_memory_entries


@dataclass(frozen=True)
class OllamaAnswer:
    text: str
    used_ollama: bool
    warning: str | None = None


class OllamaClient:
    """Tiny stdlib client for the local Ollama HTTP API."""

    def __init__(
        self,
        *,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 4.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request_json(self, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def availability_warning(self) -> str | None:
        try:
            data = self._request_json("/api/tags")
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            return (
                "Ollama is unavailable; falling back to synthetic/template answers "
                f"({exc})."
            )

        models = data.get("models", [])
        names = {
            str(item.get("name", "")).split(":", maxsplit=1)[0]
            for item in models
            if isinstance(item, dict)
        }
        full_names = {
            str(item.get("name", ""))
            for item in models
            if isinstance(item, dict)
        }
        if self.model not in names and self.model not in full_names:
            return (
                f"Ollama model '{self.model}' is not installed; falling back to "
                f"synthetic/template answers. Run `ollama pull {self.model}` to install it."
            )
        return None

    def generate_final_answer(
        self,
        task: SyntheticTask,
        trajectory: AgentTrajectory,
    ) -> OllamaAnswer:
        fallback = synthetic_final_answer(task, trajectory)
        warning = self.availability_warning()
        if warning is not None:
            return OllamaAnswer(text=fallback, used_ollama=False, warning=warning)

        tool_lines = [
            f"- {tool_call.tool_name}: {tool_call.output}"
            for tool_call in trajectory.tool_outputs
        ]
        memory_text = "\n".join(
            f"- {key}: {value}" for key, value in visible_memory_entries(task).items()
        )
        prompt = "\n".join(
            [
                "You are a local open-source model generating a concise final answer.",
                "Use only the provided synthetic task, selected actions, tool outputs, and memory.",
                "Do not claim live internet access.",
                "",
                f"User task: {task.prompt}",
                f"Task type: {task.task_type}",
                f"Selected actions: {', '.join(trajectory.actions)}",
                "Tool outputs:",
                "\n".join(tool_lines) if tool_lines else "- none",
                "Memory:",
                memory_text if memory_text else "- none",
                "",
                "Answer concisely.",
            ]
        )
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        try:
            data = self._request_json("/api/generate", payload)
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            return OllamaAnswer(
                text=fallback,
                used_ollama=False,
                warning=(
                    "Ollama generation failed; falling back to synthetic/template answers "
                    f"({exc})."
                ),
            )

        text = str(data.get("response", "")).strip()
        if not text:
            return OllamaAnswer(
                text=fallback,
                used_ollama=False,
                warning="Ollama returned an empty response; falling back to synthetic/template answers.",
            )
        return OllamaAnswer(text=text, used_ollama=True)
