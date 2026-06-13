import re
import json
import sys
from pathlib import Path

# Add root path to sys.path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from dotenv import load_dotenv
load_dotenv()

from main import initialize_rag
from rag.retriever import retrieve_documents

def parse_log():
    log_path = Path(r"C:\Users\monoj\.gemini\antigravity\brain\5a596ad9-fe37-48ba-983c-e67ed87c8a6b\.system_generated\tasks\task-580.log")
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all question blocks
    pattern = r"\[(\d+)/50\] Question: (.*?)\nExpanded queries: (.*?)\nAnswer: (.*?)\n"
    matches = re.findall(pattern, content, re.DOTALL)
    
    parsed = []
    for num, question, expanded_str, answer in matches:
        # Evaluate the python list string representation of expanded queries
        try:
            expanded = eval(expanded_str.strip())
        except Exception:
            expanded = []
        parsed.append({
            "number": int(num),
            "question": question.strip(),
            "expanded": expanded,
            "answer": answer.strip()
        })
    return parsed

def main():
    print("Parsing task-580.log...")
    parsed_items = parse_log()
    print(f"Extracted {len(parsed_items)} questions and answers from log.")

    print("Initializing RAG pipeline for retrieval...")
    collection = initialize_rag()

    with open(root_path / "eval" / "eval_set.json") as f:
        eval_set = json.load(f)

    # Map question to its ground truth from eval_set
    gt_map = {item["question"]: item.get("answer", "") for item in eval_set}

    results = []
    for item in parsed_items:
        q = item["question"]
        ans = item["answer"]
        exp = item["expanded"]
        print(f"Retrieving context for Q{item['number']}: {q[:60]}...")
        contexts = retrieve_documents(collection, q, exp)
        
        results.append({
            "question": q,
            "ground_truth": gt_map.get(q, ""),
            "answer": ans,
            "contexts": contexts
        })

    output_path = root_path / "eval" / "results_expanded.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Successfully restored {len(results)} items to results_expanded.json")

if __name__ == "__main__":
    main()
