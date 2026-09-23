# from openai import OpenAI
from ollama import Client
from config import MODEL, PERSONAS


class Agent:
    def __init__(self, name: str, client: Client):
        self.name = name
        self.persona = PERSONAS[name]
        self.client = client

    def _call(self, instructions: str, prompt: str) -> str:
        response = self.client.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": instructions,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            # instructions=instructions,
            # input=prompt,
        )

        return response["message"]["content"].strip()

    def initial_response(self, question: str) -> str:
        """
        Generate the agent's independent initial response.

        IMPORTANT:
        The agent receives:
        - its persona
        - the question

        It does NOT receive any other agent's response.
        """

        instructions = f"""
You are Agent {self.name} in a three-agent simulation.

Your persona:

{self.persona}

You are now in the INITIAL RESPONSE phase.

You must independently answer the question.

IMPORTANT:
- You have not seen any other agent's response.
- Do not assume what the other agents will say.
- Your answer must be based only on the question and your persona.
- You may give any position that follows from your reasoning.
- Begin your response with exactly YES or NO.
- After YES or NO, provide a concise explanation.
"""

        prompt = f"""
Original question:

{question}

Give your independent initial answer.
"""

        return self._call(instructions, prompt)

    def discussion_response(
        self,
        question: str,
        initial_responses: dict,
        discussion_history: list,
    ) -> str:
        """
        Generate one discussion response.

        The agent receives:
        - its persona
        - the original question
        - every agent's initial response
        - every discussion response that occurred previously
        """

        initial_context = "\n\n".join(
            f"Agent {agent_name}'s initial response:\n{response}"
            for agent_name, response in initial_responses.items()
        )

        if discussion_history:
            discussion_context = "\n\n".join(
                f"Turn {entry['turn']} - Agent {entry['agent']}:\n"
                f"{entry['response']}"
                for entry in discussion_history
            )
        else:
            discussion_context = "(No discussion responses yet.)"

        instructions = f"""
You are Agent {self.name} in a three-agent discussion.

Your persona:

{self.persona}

You are now in the DISCUSSION phase.

You have complete access to the discussion context.

You know:
1. The original question.
2. Your own initial response.
3. The initial responses of Agents A, B, and C.
4. Every discussion response that occurred before your current turn.

Your task is to participate naturally in the discussion.

You should:
- Directly engage with the arguments made by the other agents.
- Identify agreements and disagreements.
- Explain why you agree or disagree.
- Reconsider your position when another agent presents a convincing argument.
- Defend your position when you believe another argument is weak.
- Build on useful arguments from previous turns.
- Avoid pretending that another agent said something they did not say.
- Maintain your persona throughout the discussion.
- You are allowed to change your position.

Do not simply repeat your previous response. Each turn should meaningfully
respond to the current state of the discussion.

You do NOT need to begin with YES or NO during the discussion. Focus on
reasoning and dialogue.
"""

        prompt = f"""
ORIGINAL QUESTION

{question}


INITIAL RESPONSES

{initial_context}


DISCUSSION SO FAR

{discussion_context}


It is now Agent {self.name}'s turn.

Respond to the discussion.
"""

        return self._call(instructions, prompt)

    def final_response(
        self,
        question: str,
        initial_responses: dict,
        discussion_history: list,
    ) -> str:
        """
        Generate the agent's final response after the discussion.
        """

        initial_context = "\n\n".join(
            f"Agent {agent_name}'s initial response:\n{response}"
            for agent_name, response in initial_responses.items()
        )

        discussion_context = "\n\n".join(
            f"Turn {entry['turn']} - Agent {entry['agent']}:\n"
            f"{entry['response']}"
            for entry in discussion_history
        )

        instructions = f"""
You are Agent {self.name} in a three-agent simulation.

Your persona:

{self.persona}

The discussion phase has now ended.

You have access to:
1. The original question.
2. All three agents' initial responses.
3. The complete discussion transcript.

Now provide your FINAL answer to the original question.

Your final answer should only be YES or NO. 

IMPORTANT:
- You are allowed to keep your original position.
- You are allowed to change your original position.
- Your response must be exactly YES or NO.
- Do not include any additional words. 
"""

        prompt = f"""
ORIGINAL QUESTION

{question}


INITIAL RESPONSES

{initial_context}


COMPLETE DISCUSSION

{discussion_context}


Now give your final answer to the original question.
"""

        return self._call(instructions, prompt)