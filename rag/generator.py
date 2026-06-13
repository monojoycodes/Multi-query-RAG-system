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
def _generate_answer_stream_api(prompt):
    response_stream = client.models.generate_content_stream(
        model="gemma-4-31b-it",
        contents=prompt,
    )
    # Consume stream fully here so that mid-stream 500/503 errors trigger retries
    return list(response_stream)


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

    response_stream = _generate_answer_stream_api(prompt)

    buffer = ""
    seen_start = False

    for chunk in response_stream:
        chunk_text = ""
        if chunk.candidates and chunk.candidates[0].content.parts:
            for part in chunk.candidates[0].content.parts:
                if getattr(part, "thought", False):
                    continue
                if part.text:
                    chunk_text += part.text
        
        if chunk_text:
            buffer += chunk_text
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