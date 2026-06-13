import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add root path to sys.path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.outputs import ChatResult

# Custom wrapper to intercept multi-candidate requests and simulate them sequentially.
# This prevents "Multiple candidates is not enabled for this model" errors on models like gemma.
class SafeChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Inspect model attributes and parameters
        n = getattr(self, "n", 1) or 1
        candidate_count = getattr(self, "candidate_count", 1) or 1
        
        n = kwargs.get("n", n)
        candidate_count = kwargs.get("candidate_count", candidate_count)
        
        gen_config = kwargs.get("generation_config", {})
        if isinstance(gen_config, dict):
            candidate_count = gen_config.get("candidate_count", candidate_count)
            
        candidates_to_generate = max(n, candidate_count)
        
        if candidates_to_generate > 1:
            print(f"\n[Wrapper] Intercepted multi-candidate request (n={candidates_to_generate}). Simulating via sequential calls...")
            
            # Temporarily set candidate_count to 1 on self
            orig_candidate = self.candidate_count
            self.candidate_count = 1
            
            single_kwargs = kwargs.copy()
            if "n" in single_kwargs:
                single_kwargs["n"] = 1
            if "candidate_count" in single_kwargs:
                single_kwargs["candidate_count"] = 1
            if "generation_config" in single_kwargs and isinstance(single_kwargs["generation_config"], dict):
                single_kwargs["generation_config"] = single_kwargs["generation_config"].copy()
                single_kwargs["generation_config"]["candidate_count"] = 1
                
            generations = []
            try:
                for i in range(candidates_to_generate):
                    result = super()._generate(messages, stop=stop, run_manager=run_manager, **single_kwargs)
                    generations.extend(result.generations)
            finally:
                self.candidate_count = orig_candidate
                
            return ChatResult(generations=generations)
        else:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # Inspect model attributes and parameters
        n = getattr(self, "n", 1) or 1
        candidate_count = getattr(self, "candidate_count", 1) or 1
        
        n = kwargs.get("n", n)
        candidate_count = kwargs.get("candidate_count", candidate_count)
        
        gen_config = kwargs.get("generation_config", {})
        if isinstance(gen_config, dict):
            candidate_count = gen_config.get("candidate_count", candidate_count)
            
        candidates_to_generate = max(n, candidate_count)
        
        if candidates_to_generate > 1:
            print(f"\n[Wrapper Async] Intercepted multi-candidate request (n={candidates_to_generate}). Simulating via sequential calls...")
            
            # Temporarily set candidate_count to 1 on self
            orig_candidate = self.candidate_count
            self.candidate_count = 1
            
            single_kwargs = kwargs.copy()
            if "n" in single_kwargs:
                single_kwargs["n"] = 1
            if "candidate_count" in single_kwargs:
                single_kwargs["candidate_count"] = 1
            if "generation_config" in single_kwargs and isinstance(single_kwargs["generation_config"], dict):
                single_kwargs["generation_config"] = single_kwargs["generation_config"].copy()
                single_kwargs["generation_config"]["candidate_count"] = 1
                
            generations = []
            try:
                for i in range(candidates_to_generate):
                    result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **single_kwargs)
                    generations.extend(result.generations)
            finally:
                self.candidate_count = orig_candidate
                
            return ChatResult(generations=generations)
        else:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

# 1. Initialize models
print("Initializing evaluator LLM and Embeddings...")
gemini_llm = SafeChatGoogleGenerativeAI(model="gemma-4-26b-a4b-it", timeout=60)
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

evaluator_llm = LangchainLLMWrapper(gemini_llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

# 2. Configure metrics with models
faithfulness.llm = evaluator_llm
answer_relevancy.llm = evaluator_llm
answer_relevancy.embeddings = evaluator_embeddings
context_precision.llm = evaluator_llm
context_precision.embeddings = evaluator_embeddings

metrics = [faithfulness, answer_relevancy, context_precision]

def score_file(filepath):
    print(f"\nScoring {filepath.name}...")
    if not filepath.exists():
        print(f"Error: {filepath} does not exist.")
        return None
        
    with open(filepath) as f:
        results = json.load(f)
        
    data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results]
    }
    
    dataset = Dataset.from_dict(data)
    
    # Run evaluation
    run_config = RunConfig(max_workers=2, timeout=60, max_retries=2)
    score_result = evaluate(dataset, metrics=metrics, run_config=run_config)
    return dict(score_result)

def main():
    eval_dir = root_path / "eval"
    baseline_path = eval_dir / "results_baseline.json"
    expanded_path = eval_dir / "results_expanded.json"
    
    scores = {}
    
    if baseline_path.exists():
        scores["Baseline (No Expansion)"] = score_file(baseline_path)
    
    if expanded_path.exists():
        scores["Expanded (With Expansion)"] = score_file(expanded_path)
        
    print("\n" + "="*50)
    print("EVALUATION RESULTS COMPARISON")
    print("="*50)
    
    for name, score_dict in scores.items():
        if score_dict is None:
            continue
        print(f"\nConfiguration: {name}")
        for metric, score in score_dict.items():
            print(f"  {metric}: {score:.4f}")
            
    if "Baseline (No Expansion)" in scores and "Expanded (With Expansion)" in scores:
        b = scores["Baseline (No Expansion)"]
        e = scores["Expanded (With Expansion)"]
        if b is not None and e is not None:
            print("\n" + "="*50)
            print("A/B DELTA COMPARISON")
            print("="*50)
            for metric in b.keys():
                delta = e[metric] - b[metric]
                print(f"  {metric}: Baseline={b[metric]:.4f} | Expanded={e[metric]:.4f} | Delta={delta:+.4f}")

if __name__ == "__main__":
    main()
