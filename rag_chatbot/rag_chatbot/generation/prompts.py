"""
generation/prompts.py
---------------------
System prompt templates for grounded RAG generation.

Key requirements enforced by the prompt:
  1. Answer ONLY using the provided context passages.
  2. Cite the source document by name for every factual claim.
  3. Respond with a specific refusal message if the answer cannot be found.
"""

SYSTEM_PROMPT = """\
You are a knowledgeable and precise research assistant.
You answer questions EXCLUSIVELY based on the context passages provided below.
You do NOT use any prior knowledge or information outside of these passages.

## Rules you MUST follow

1. **Ground every factual claim** in one or more of the provided context passages.
2. **Cite sources explicitly** — after each factual sentence, add a parenthetical
   citation with the source name shown in the passage header, e.g. (Source: Report Title).
3. **Never fabricate** information. If the passages do not contain the answer, say EXACTLY:
   "I cannot find this in the provided documents."
4. **Do not speculate** or fill gaps with general knowledge.
5. Keep your answer concise and directly responsive to the question.

## Context passages

{context}

## End of context

Remember: answer only from the passages above, cite every claim, and say
"I cannot find this in the provided documents." if the answer is absent.
"""

REFUSAL_PHRASE = "I cannot find this in the provided documents."


def build_system_prompt(context: str) -> str:
    """Render the system prompt with the retrieved context injected."""
    return SYSTEM_PROMPT.format(context=context)


def build_messages(query: str, context: str, history: list | None = None):
    """
    Build the messages list for the LLM API call.

    history: list of (user_msg, assistant_msg) tuples
    """
    system = build_system_prompt(context)
    messages = []

    if history:
        for user_turn, assistant_turn in history:
            messages.append({"role": "user", "content": user_turn})
            messages.append({"role": "assistant", "content": assistant_turn})

    messages.append({"role": "user", "content": query})
    return system, messages
