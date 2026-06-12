from google import genai
import re

client = genai.Client()


def generate_answer(
    question,
    documents
):

    context = "\n\n".join(documents)

    prompt = f"""Answer the question using only
the provided context. If the context is not there then simply say "I do not know".

Context:
{context}

Question:
{question}

Provide your final answer wrapped inside <answer>...</answer> XML tags. Do not put any thinking, chain-of-thought, or other content inside the XML tags.
"""

    response_stream = client.models.generate_content_stream(
        model="gemma-4-26b-a4b-it",
        contents=prompt
    )

    buffer = ""
    seen_start = False

    for chunk in response_stream:
        if chunk.text:
            buffer += chunk.text
            if not seen_start:
                if "<answer>" in buffer:
                    parts = buffer.split("<answer>", 1)
                    buffer = parts[1]
                    seen_start = True
            
            if seen_start:
                if "</answer>" in buffer:
                    final_parts = buffer.split("</answer>", 1)
                    yield final_parts[0]
                    break
                else:
                    if len(buffer) > 9:
                        yield buffer[:-9]
                        buffer = buffer[-9:]

    if not seen_start and buffer:
        yield buffer.strip()