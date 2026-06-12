def split_text_recursive(text, separators, chunk_size, chunk_overlap):
    if not separators:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

    separator = separators[0]
    next_separators = separators[1:]

    parts = text.split(separator)
    chunks = []
    current_chunk = ""

    for part in parts:
        if len(current_chunk) + len(part) + (len(separator) if current_chunk else 0) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = current_chunk[overlap_start:]

            if len(part) > chunk_size:
                sub_chunks = split_text_recursive(part, next_separators, chunk_size, chunk_overlap)
                for sc in sub_chunks:
                    if len(current_chunk) + len(sc) + (len(separator) if current_chunk else 0) > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                            overlap_start = max(0, len(current_chunk) - chunk_overlap)
                            current_chunk = current_chunk[overlap_start:]
                        current_chunk = (current_chunk + separator + sc) if current_chunk else sc
                    else:
                        current_chunk = (current_chunk + separator + sc) if current_chunk else sc
            else:
                current_chunk = (current_chunk + separator + part) if current_chunk else part
        else:
            current_chunk = (current_chunk + separator + part) if current_chunk else part

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def split_documents(texts):
    """
    Split PDF text into chunks suitable for embeddings.
    Uses a pure Python recursive character splitter to avoid heavy external dependencies.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunk_size = 1000
    chunk_overlap = 100

    full_text = "\n\n".join(texts)
    return split_text_recursive(full_text, separators, chunk_size, chunk_overlap)