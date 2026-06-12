from rag.loader import load_pdf
from rag.splitter import split_documents
from rag.vectordb import create_vector_store
from rag.query_expansion import generate_queries
from rag.retriever import retrieve_documents
from rag.generator import generate_answer


def run_rag(collection, question):
    """
    Executes the full RAG pipeline:
    1. Query expansion (generating alternative search queries).
    2. Retrieval (fetching relevant documents for all queries).
    3. Generation (answering the question using retrieved documents).
    """
    # 1. Expand queries
    expanded_queries = generate_queries(question)
    print("\nGenerated Multi-Queries for Expansion:")
    for idx, q in enumerate(expanded_queries):
        print(f"  {idx + 1}. {q}")

    # 2. Retrieve documents
    documents = retrieve_documents(collection, question, expanded_queries)

    # 3. Generate answer
    answer = generate_answer(question, documents)

    return answer


def main():

    print("Loading PDF...")

    texts = load_pdf(
        "data/google-2025-environmental-report.pdf"
    )

    print(f"Loaded {len(texts)} pages")

    print("Splitting documents...")

    chunks = split_documents(texts)

    print(f"Created {len(chunks)} chunks")

    print("Creating vector database...")

    collection = create_vector_store(chunks)

    print("Ready!")

    while True:

        question = input("\nAsk a question (or type exit): ")

        if question.lower() == "exit":
            break

        answer = run_rag(
            collection,
            question
        )

        print("\nAnswer:")
        for chunk in answer:
            print(chunk, end="", flush=True)
        print()
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()