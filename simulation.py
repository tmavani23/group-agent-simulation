# simulation.py

import json
import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from agents import Agent
from config import (
    AGENT_ORDER,
    DISCUSSION_ROUNDS,
    MODEL,
    PERSONAS,
    QUESTION,
)


def create_agents(client):
    """
    Create the three agents.
    """

    return {
        agent_name: Agent(agent_name, client)
        for agent_name in AGENT_ORDER
    }


def run_initial_phase(agents):
    """
    Each agent independently answers the question.

    No agent sees another agent's response during this phase.
    """

    print("\n" + "=" * 70)
    print("INITIAL RESPONSES")
    print("=" * 70)

    initial_responses = {}

    for agent_name in AGENT_ORDER:
        print(f"\nAgent {agent_name} is answering independently...")

        response = agents[agent_name].initial_response(
            QUESTION
        )

        initial_responses[agent_name] = response

        print(f"\nAgent {agent_name}:")
        print(response)

    return initial_responses


def run_discussion_phase(agents, initial_responses):
    """
    Run exactly:

        A -> B -> C
        A -> B -> C
        A -> B -> C
        A -> B -> C

    Every agent sees:
    - the question
    - all initial responses
    - all previous discussion responses
    """

    print("\n" + "=" * 70)
    print("DISCUSSION")
    print("=" * 70)

    discussion_history = []

    turn_number = 1

    for round_number in range(1, DISCUSSION_ROUNDS + 1):

        print(f"\n--- Discussion Round {round_number} ---")

        for agent_name in AGENT_ORDER:

            print(
                f"\nAgent {agent_name} is thinking "
                f"(turn {turn_number})..."
            )

            response = agents[agent_name].discussion_response(
                question=QUESTION,
                initial_responses=initial_responses,
                discussion_history=discussion_history,
            )

            entry = {
                "turn": turn_number,
                "round": round_number,
                "agent": agent_name,
                "response": response,
            }

            discussion_history.append(entry)

            print(f"\nAgent {agent_name}:")
            print(response)

            turn_number += 1

    return discussion_history


def run_final_phase(
    agents,
    initial_responses,
    discussion_history,
):
    """
    After the discussion, ask every agent the original question again.
    """

    print("\n" + "=" * 70)
    print("FINAL RESPONSES")
    print("=" * 70)

    final_responses = {}

    for agent_name in AGENT_ORDER:

        print(
            f"\nAgent {agent_name} is giving their final answer..."
        )

        response = agents[agent_name].final_response(
            question=QUESTION,
            initial_responses=initial_responses,
            discussion_history=discussion_history,
        )

        final_responses[agent_name] = response

        print(f"\nAgent {agent_name}:")
        print(response)

    return final_responses


def save_results(
    initial_responses,
    discussion_history,
    final_responses,
):
    """
    Save the complete experiment to JSON.
    """

    os.makedirs("results", exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"results/simulation_{timestamp}.json"
    )

    results = {
        "metadata": {
            "timestamp": timestamp,
            "model": MODEL,
            "discussion_rounds": DISCUSSION_ROUNDS,
            "discussion_order": (
                AGENT_ORDER * DISCUSSION_ROUNDS
            ),
        },

        "question": QUESTION,

        "personas": PERSONAS,

        "initial_responses": initial_responses,

        "discussion": discussion_history,

        "final_responses": final_responses,
    }

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return filename


def print_summary(
    initial_responses,
    final_responses,
):
    """
    Print a simple initial vs. final comparison.
    """

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for agent_name in AGENT_ORDER:

        initial = initial_responses[agent_name]
        final = final_responses[agent_name]

        print(f"\nAgent {agent_name}")
        print("-" * 40)

        print("Initial:")
        print(initial)

        print("\nFinal:")
        print(final)


def main():

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY was not found. "
            "Make sure it is set in your .env file."
        )

    client = OpenAI(api_key=api_key)

    agents = create_agents(client)

    # --------------------------------------------------------
    # Phase 1
    # --------------------------------------------------------

    initial_responses = run_initial_phase(
        agents
    )

    # --------------------------------------------------------
    # Phase 2
    # --------------------------------------------------------

    discussion_history = run_discussion_phase(
        agents,
        initial_responses,
    )

    # --------------------------------------------------------
    # Phase 3
    # --------------------------------------------------------

    final_responses = run_final_phase(
        agents,
        initial_responses,
        discussion_history,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    filename = save_results(
        initial_responses,
        discussion_history,
        final_responses,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        initial_responses,
        final_responses,
    )

    print("\n" + "=" * 70)
    print(f"Complete simulation saved to: {filename}")
    print("=" * 70)


if __name__ == "__main__":
    main()