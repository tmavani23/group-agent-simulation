import json
from datetime import datetime
from pathlib import Path

from ollama import Client

from agents import Agent
from config import AGENT_ORDER, DISCUSSION_ROUNDS, MODEL, PERSONAS, QUESTION

from .conversation_state import ConversationState
from .orchestrator import Orchestrator


def run():
    client = Client(host="http://localhost:11434")

    agents = {
        name: Agent(name, client)
        for name in AGENT_ORDER
    }

    # Phase 1: Independent initial answers
    print("=== INDEPENDENT INITIAL ANSWERS ===")

    initial = {}

    for name in AGENT_ORDER:
        initial[name] = agents[name].initial_response(QUESTION)
        print(f"Agent {name}: {initial[name]}\n")

    # Phase 2: Natural conversation
    state = ConversationState(
        question=QUESTION,
        initial_responses=initial,
        agent_order=list(AGENT_ORDER),
        turns_per_agent=DISCUSSION_ROUNDS,
    )

    orchestrator = Orchestrator(client)

    print("=== NATURAL DISCUSSION ===")

    while not state.complete:
        decision = orchestrator.select(state)
        speaker = decision["speaker"]

        response = agents[speaker].discussion_response(
            question=QUESTION,
            initial_responses=initial,
            discussion_history=list(state.history),
        )

        state.add_turn(speaker, response, decision)

        print(
            f"Turn {len(state.history)} - "
            f"Agent {speaker} [{decision['signal']}]:"
        )
        print(f"{response}\n")

    # Phase 3: Final answers
    print("=== FINAL ANSWERS ===")

    final = {}

    for name in AGENT_ORDER:
        final[name] = agents[name].final_response(
            QUESTION,
            initial,
            list(state.history),
        )

        print(f"Agent {name}: {final[name]}")

    # Save results
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL,
            "orchestrator_model": MODEL,
            "mode": "natural_conversation",
            "turns_per_agent": DISCUSSION_ROUNDS,
            "actual_order": [
                t["agent"] for t in state.history
            ],
            "counts": state.counts,
        },
        "question": QUESTION,
        "personas": PERSONAS,
        "initial_responses": initial,
        "discussion": state.history,
        "selection_log": state.selection_log,
        "final_responses": final,
    }

    out = Path("results/natural")
    out.mkdir(parents=True, exist_ok=True)

    path = out / (
        f"simulation_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    )

    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved to {path}")

    return results


if __name__ == "__main__":
    run()