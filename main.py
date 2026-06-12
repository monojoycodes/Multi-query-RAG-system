import os
import warnings
# Suppress Hugging Face and other library warnings
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from rag.loader import load_pdf
from rag.splitter import split_documents
from rag.vectordb import create_vector_store
from rag.pipeline import run_rag


PDF_PATH = "data/google-2025-environmental-report.pdf"


def initialize_rag():

    print("\nInitializing RAG Assistant...")
    print("-" * 50)

    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print("Loading PDF...")

    texts = load_pdf(PDF_PATH)

    print(f"Loaded {len(texts)} pages")

    print("Splitting documents into chunks...")

    chunks = split_documents(texts)

    print(f"Created {len(chunks)} chunks")

    print("Creating vector database...")

    collection = create_vector_store(chunks)

    print("Vector database ready")

    return collection


def chat_loop(collection):

    print("\nRAG Assistant Ready")
    print("Type 'exit' to quit")
    print("-" * 50)

    while True:

        question = input("\nQuestion: ").strip()

        if not question:
            continue

        if question.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        try:

            print("\nSearching documents...")

            answer = run_rag(
                collection=collection,
                question=question
            )

            print("\nAnswer:")
            print("-" * 50)
            for chunk in answer:
                print(chunk, end="", flush=True)
            print()

        except Exception as e:
            print(f"\nError: {e}")

        print("\n" + "=" * 80)


def main():

    try:

        collection = initialize_rag()

        chat_loop(collection)

    except KeyboardInterrupt:

        print("\n\nProgram stopped.")

    except Exception as e:

        print(f"\nStartup Error: {e}")


if __name__ == "__main__":
    main()