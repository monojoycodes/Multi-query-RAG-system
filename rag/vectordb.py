import chromadb
from rag.embeddings import embedding_function


def create_vector_store(chunks):

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        "pdf-to-rag",
        embedding_function=embedding_function
    )

    ids = [str(i) for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=chunks
    )

    return collection