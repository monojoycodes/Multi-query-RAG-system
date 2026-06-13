from google import genai
from google.genai import types
import re
import httpx

client = genai.Client(
    http_options=types.HttpOptions(
        httpx_client=httpx.Client(timeout=120.0),
        httpx_async_client=httpx.AsyncClient(timeout=120.0)
    )
)

from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def _generate_queries_api(prompt):
    return client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt,
    )

def generate_queries(question):

    prompt = f"""Generate 5 alternative search queries to the question.

Question:
{question}

Return the queries wrapped inside <queries>...</queries> XML tags, with one query per line. Do not put any thinking or other content inside the XML tags.
"""

    response = _generate_queries_api(prompt)

    # Extract only non-thought text parts
    text_parts = []
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False):
                continue
            if part.text:
                text_parts.append(part.text)
    text = "".join(text_parts)

    match = re.search(r"<queries>(.*?)</queries>", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return [q.strip() for q in content.split("\n") if q.strip()]

    return [
        q.strip()
        for q in text.split("\n")
        if q.strip()
    ]