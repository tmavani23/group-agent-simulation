from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    question: str
    initial_responses: dict[str, str]
    agent_order: list[str]
    turns_per_agent: int = 4
    history: list[dict[str, Any]] = field(default_factory=list)
    selection_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {
            name: sum(t["agent"] == name for t in self.history)
            for name in self.agent_order
        }

    @property
    def eligible(self) -> list[str]:
        counts = self.counts
        return [
            name for name in self.agent_order
            if counts[name] < self.turns_per_agent
        ]

    @property
    def last_speaker(self) -> str | None:
        return self.history[-1]["agent"] if self.history else None

    @property
    def complete(self) -> bool:
        return not self.eligible

    def add_turn(self, agent: str, response: str, selection: dict[str, Any]) -> None:
        if agent not in self.eligible:
            raise ValueError(f"{agent} has exhausted its speaking budget")

        turn = len(self.history) + 1

        self.selection_log.append({
            "before_turn": turn,
            "counts_before": self.counts.copy(),
            **selection,
        })

        self.history.append({
            "turn": turn,
            "agent": agent,
            "response": response,
        })