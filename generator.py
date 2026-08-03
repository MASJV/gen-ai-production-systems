"""
generator.py

Stage: "OpenAI GPT-4.1 mini" -> "Final Answer"

This is the last step. We take the best chunks found so far, put
them into a prompt as "context", and ask the chat model to answer
the user's question using only that context.
"""

from openai import OpenAI

import config


def build_context_text(chunks):
    """
    Joins the text of all chunks into one big string, and labels
    each piece with the file it came from.
    """
    context_parts = []
    for chunk_info in chunks:
        part = "Source: " + chunk_info["source"] + "\n" + chunk_info["text"]
        context_parts.append(part)

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text


def generate_answer(question, chunks, api_key):
    """
    Sends the question and the context chunks to the chat model
    and returns the final answer as plain text.
    """
    client = OpenAI(api_key=api_key)

    context_text = build_context_text(chunks)

    system_message = (
        "You are an enterprise document assistant. "
        "Answer the question using only the given context. "
        "If the answer is not in the context, say you do not know."
    )

    user_message = "Context:\n" + context_text + "\n\nQuestion: " + question

    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    return answer