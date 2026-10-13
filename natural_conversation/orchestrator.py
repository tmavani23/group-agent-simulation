import json
import re

from ollama import Client
from config import MODEL
from .conversation_state import ConversationState


class Orchestrator:
    def __init__(self, client: Client, model: str = MODEL):
        self.client = client
        self.model = model

    def _evaluate(self, state: ConversationState, candidates: list[str]) -> dict:
        initial = "\n\n".join(
            f"Agent {a}: {r}"
            for a, r in state.initial_responses.items()
        )

        history = "\n\n".join(
            f"Turn {t['turn']} - Agent {t['agent']}: {t['response']}"
            for t in state.history
        ) or "(No discussion turns yet)"

        prompt = f"""ORIGINAL QUESTION:
{state.question}

INITIAL INDEPENDENT RESPONSES:
{initial}

DISCUSSION SO FAR:
{history}

ELIGIBLE SPEAKERS: {', '.join(candidates)}
LAST SPEAKER: {state.last_speaker or 'none'}
SPEAKING COUNTS: {json.dumps(state.counts)}

Evaluate EACH eligible speaker's conversational opportunity.

Prefer:
1. Direct questions addressed to an agent.
2. Direct challenges to an agent's previous claims.
3. Unresolved arguments an agent can address.
4. Relevant new perspectives.

Do not reward verbosity, agreement, or a particular position.
Do not generate a discussion response.

Return ONLY a JSON object in this structure:
{{
  "candidates": [
    {{
      "agent": "A",
      "score": 0,
      "signal": "direct_question|direct_challenge|unresolved_argument|new_perspective|opening",
      "reason": "short factual reason"
    }}
  ]
}}

Include exactly one entry for EACH eligible speaker.

Scores:
4 = direct question
3 = direct challenge
2 = unresolved argument
1 = new perspective
0 = no specific opening

Only assign 4 if explicitly addressed or asked.
Only assign 3 if their earlier claim was explicitly challenged.
"""

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a neutral conversation floor manager, "
                        "not a participant. Output valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0},
        )

        raw = response["message"]["content"]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.S)
            if not match:
                raise ValueError(f"Invalid orchestrator JSON: {raw}")
            data = json.loads(match.group())

        if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
            raise ValueError(f"Invalid candidate structure: {data}")

        return data

    def select(self, state: ConversationState) -> dict:
        eligible = state.eligible

        if not eligible:
            raise ValueError("Discussion is complete")

        # Avoid consecutive turns by the same agent when possible.
        candidates = [
            a for a in eligible
            if a != state.last_speaker
        ] or eligible

        evaluation = self._evaluate(state, candidates)

        validated = {}

        for item in evaluation["candidates"]:
            if not isinstance(item, dict):
                continue

            name = item.get("agent")

            if name not in candidates or name in validated:
                continue

            try:
                score = int(item.get("score", 0))
            except (TypeError, ValueError):
                score = 0

            validated[name] = {
                "agent": name,
                "score": max(0, min(4, score)),
                "signal": str(item.get("signal", "opening")),
                "reason": str(item.get("reason", "No specific opportunity identified")),
            }

        for name in candidates:
            validated.setdefault(name, {
                "agent": name,
                "score": 0,
                "signal": "opening",
                "reason": "No evaluated opportunity",
            })

        # Conversational relevance first, speaking balance second.
        # Original agent order is the final deterministic tie-breaker.
        ranked = sorted(
            validated.values(),
            key=lambda c: (
                -c["score"],
                state.counts[c["agent"]],
                state.agent_order.index(c["agent"]),
            ),
        )

        winner = ranked[0]

        return {
            "speaker": winner["agent"],
            "signal": winner["signal"],
            "reason": winner["reason"],
            "candidates": ranked,
            "eligible": candidates,
        }