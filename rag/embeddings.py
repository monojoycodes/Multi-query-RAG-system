import os
os.environ["HF_HUB_VERBOSITY"] = "error"

from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)