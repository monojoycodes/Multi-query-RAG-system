import json
import sys
import os
from pathlib import Path

# Add root path to sys.path to allow importing from main and rag modules
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from dotenv import load_dotenv
load_dotenv()

from main import initialize_rag
from rag.query_expansion import generate_queries
from rag.retriever import retrieve_documents
from rag.generator import generate_answer

# Toggle this for A/B testing
USE_EXPANSION = False

def main():
    print(f"Initializing RAG pipeline (USE_EXPANSION={USE_EXPANSION})...")
    collection = initialize_rag()

    with open(root_path / "eval" / "eval_set.json") as f:
        eval_set = json.load(f)

    output_filename = "results_expanded.json" if USE_EXPANSION else "results_baseline.json"
    output_path = root_path / "eval" / output_filename

    results = []
    processed_questions = {}
    if output_path.exists():
        try:
            with open(output_path) as f:
                results = json.load(f)
                processed_questions = {item["question"]: item for item in results}
                print(f"Found existing results. Loaded {len(results)} processed questions.")
        except Exception as e:
            print(f"Could not load existing results: {e}. Starting fresh.")
            results = []

    for idx, item in enumerate(eval_set):
        question = item["question"]
        print(f"\n[{idx+1}/{len(eval_set)}] Question: {question}")
        
        if question in processed_questions:
            print("Question already processed. Skipping.")
            continue

        if USE_EXPANSION:
            expanded = generate_queries(question)
            print(f"Expanded queries: {expanded}")
        else:
            expanded = []
            print("Running without expansion.")

        # retrieve_documents expects original_query, expanded_queries
        contexts = retrieve_documents(collection, question, expanded)
        
        # generate_answer returns a generator of string chunks
        answer_gen = generate_answer(question, contexts)
        answer = "".join(list(answer_gen))
        print(f"Answer: {answer}")

        # Map 'answer' key in eval_set.json to 'ground_truth'
        new_result = {
            "question": question,
            "ground_truth": item.get("answer", ""),
            "answer": answer,
            "contexts": contexts
        }
        results.append(new_result)
        processed_questions[question] = new_result

        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Also copy/save to results.json as standard results
    with open(root_path / "eval" / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {output_filename}")

if __name__ == "__main__":
    main()
