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

    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt
    )

    text = response.text
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()