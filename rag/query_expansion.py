from google import genai
import re

client = genai.Client()

def generate_queries(question):

    prompt = f"""Generate 5 alternative search queries to the question.

Question:
{question}

Return the queries wrapped inside <queries>...</queries> XML tags, with one query per line. Do not put any thinking or other content inside the XML tags.
"""

    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt
    )

    text = response.text
    match = re.search(r"<queries>(.*?)</queries>", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return [q.strip() for q in content.split("\n") if q.strip()]

    return [
        q.strip()
        for q in text.split("\n")
        if q.strip()
    ]