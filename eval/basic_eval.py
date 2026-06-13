import json
import numpy as np
import sys
from pathlib import Path

# Add root path to sys.path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from rag.embeddings import embedding_function

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def evaluate_results(filepath):
    if not filepath.exists():
        return None
        
    print(f"Loading {filepath.name}...")
    with open(filepath) as f:
        results = json.load(f)
        
    answers = [r["answer"] for r in results]
    ground_truths = [r["ground_truth"] for r in results]
    
    # Get embeddings
    print(f"Generating local embeddings for {filepath.name}...")
    ans_embeddings = embedding_function(answers)
    gt_embeddings = embedding_function(ground_truths)
    
    similarities = []
    idk_count = 0
    exact_matches = 0
    
    for ans, gt, ans_emb, gt_emb in zip(answers, ground_truths, ans_embeddings, gt_embeddings):
        # Cosine similarity
        sim = cosine_similarity(ans_emb, gt_emb)
        similarities.append(sim)
        
        # IDK rate
        if "i do not know" in ans.lower() or "i don't know" in ans.lower():
            idk_count += 1
            
        # Exact Match or Ground Truth Containment
        clean_ans = ans.lower().strip()
        clean_gt = gt.lower().strip()
        # strip citations/cites if any
        if "[cite" in clean_gt:
            clean_gt = clean_gt.split("[cite")[0].strip()
        if "[cite" in clean_ans:
            clean_ans = clean_ans.split("[cite")[0].strip()
            
        if clean_gt in clean_ans or clean_ans in clean_gt:
            exact_matches += 1
            
    return {
        "avg_semantic_similarity": float(np.mean(similarities)),
        "idk_rate": float(idk_count / len(results)),
        "exact_match_rate": float(exact_matches / len(results)),
        "raw_results": results,
        "similarities": similarities
    }

def generate_html_report(baseline, expanded, output_path):
    print("Generating HTML comparison report...")
    
    # Build list of questions side-by-side
    question_cards = ""
    for idx, (b_item, e_item) in enumerate(zip(baseline["raw_results"], expanded["raw_results"])):
        q = b_item["question"]
        gt = b_item["ground_truth"]
        b_ans = b_item["answer"]
        e_ans = e_item["answer"]
        
        b_sim = baseline["similarities"][idx]
        e_sim = expanded["similarities"][idx]
        
        # Highlight color for higher similarity
        b_badge_class = "better" if b_sim > e_sim else ""
        e_badge_class = "better" if e_sim > b_sim else ""
        
        if abs(b_sim - e_sim) < 1e-4:
            b_badge_class = e_badge_class = "tie"
            
        question_cards += f"""
        <div class="question-card">
            <div class="question-header">
                <span class="question-num">Q{idx+1}</span>
                <span class="question-text">{q}</span>
            </div>
            <div class="grid-3">
                <div class="column ground-truth">
                    <h4>Ground Truth (Correct)</h4>
                    <div class="ans-text">{gt}</div>
                </div>
                <div class="column baseline">
                    <h4>Single Query (Baseline) <span class="metric-badge {b_badge_class}">Sim: {b_sim:.4f}</span></h4>
                    <div class="ans-text">{b_ans}</div>
                </div>
                <div class="column expanded">
                    <h4>Multi-Query (Expanded) <span class="metric-badge {e_badge_class}">Sim: {e_sim:.4f}</span></h4>
                    <div class="ans-text">{e_ans}</div>
                </div>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG A/B Comparison Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f17;
            --card-bg: #151c2c;
            --text-color: #cbd5e1;
            --text-muted: #64748b;
            --accent-color: #3b82f6;
            --success-color: #10b981;
            --warn-color: #f59e0b;
            --danger-color: #ef4444;
            --border-color: #243049;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 3rem;
            text-align: center;
        }}
        header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            color: #fff;
        }}
        header p {{
            color: var(--text-muted);
            font-size: 1.1rem;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        .summary-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }}
        .summary-card h3 {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .summary-card .val-row {{
            display: flex;
            gap: 1.5rem;
            align-items: baseline;
            margin: 0.5rem 0;
        }}
        .summary-card .value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #fff;
        }}
        .summary-card .value.lbl {{
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
        }}
        .summary-card .delta {{
            font-size: 0.9rem;
            font-weight: 600;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            margin-top: 0.5rem;
        }}
        .delta.positive {{
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--success-color);
        }}
        .delta.negative {{
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--danger-color);
        }}
        .delta.neutral {{
            background-color: rgba(100, 116, 139, 0.15);
            color: var(--text-muted);
        }}
        .question-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .question-header {{
            display: flex;
            align-items: center;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
        }}
        .question-num {{
            background-color: var(--accent-color);
            color: #fff;
            padding: 0.25rem 0.75rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.85rem;
            margin-right: 1rem;
        }}
        .question-text {{
            font-size: 1.2rem;
            font-weight: 600;
            color: #fff;
        }}
        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
        }}
        .column {{
            background-color: rgba(11, 15, 23, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
        }}
        .column.ground-truth {{
            border-color: rgba(245, 158, 11, 0.3);
            background-color: rgba(245, 158, 11, 0.02);
        }}
        .column.baseline {{
            border-color: rgba(100, 116, 139, 0.3);
        }}
        .column.expanded {{
            border-color: rgba(59, 130, 246, 0.3);
            background-color: rgba(59, 130, 246, 0.01);
        }}
        .column h4 {{
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .column.ground-truth h4 {{ color: var(--warn-color); }}
        .column.baseline h4 {{ color: var(--text-muted); }}
        .column.expanded h4 {{ color: var(--accent-color); }}
        .ans-text {{
            font-size: 0.95rem;
            color: #cbd5e1;
            flex-grow: 1;
            white-space: pre-wrap;
        }}
        .metric-badge {{
            font-size: 0.75rem;
            font-weight: 600;
            background-color: var(--border-color);
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            color: var(--text-color);
        }}
        .metric-badge.better {{
            background-color: rgba(16, 185, 129, 0.2);
            color: var(--success-color);
        }}
        .metric-badge.tie {{
            background-color: rgba(245, 158, 11, 0.2);
            color: var(--warn-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>RAG A/B Comparison Report</h1>
            <p>Evaluation results for Baseline (No Expansion) vs. Expanded (With Expansion) pipeline</p>
        </header>

        <div class="summary-cards">
            <div class="summary-card">
                <h3>Semantic Similarity</h3>
                <div class="val-row">
                    <div><span class="value">{baseline['avg_semantic_similarity']:.4f}</span> <span class="value lbl">Base</span></div>
                    <div><span class="value">{expanded['avg_semantic_similarity']:.4f}</span> <span class="value lbl">Exp</span></div>
                </div>
                <div class="delta positive">Delta: +{(expanded['avg_semantic_similarity'] - baseline['avg_semantic_similarity']):+.4f}</div>
            </div>
            
            <div class="summary-card">
                <h3>Exact Match Rate</h3>
                <div class="val-row">
                    <div><span class="value">{baseline['exact_match_rate']*100:.1f}%</span> <span class="value lbl">Base</span></div>
                    <div><span class="value">{expanded['exact_match_rate']*100:.1f}%</span> <span class="value lbl">Exp</span></div>
                </div>
                <div class="delta positive">Delta: +{(expanded['exact_match_rate'] - baseline['exact_match_rate'])*100:+.1f}%</div>
            </div>

            <div class="summary-card">
                <h3>Retrieval Failure Rate</h3>
                <div class="val-row">
                    <div><span class="value">{baseline['idk_rate']*100:.1f}%</span> <span class="value lbl">Base</span></div>
                    <div><span class="value">{expanded['idk_rate']*100:.1f}%</span> <span class="value lbl">Exp</span></div>
                </div>
                <div class="delta negative">Delta: {(expanded['idk_rate'] - baseline['idk_rate'])*100:+.1f}%</div>
            </div>
        </div>

        <div class="questions-section">
            {question_cards}
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML comparison report written to {output_path.name}")

def main():
    eval_dir = root_path / "eval"
    baseline_path = eval_dir / "results_baseline.json"
    expanded_path = eval_dir / "results_expanded.json"
    
    baseline_metrics = evaluate_results(baseline_path)
    expanded_metrics = evaluate_results(expanded_path)
    
    print("\n" + "="*50)
    print("BASIC LOCAL EVALUATION COMPARISON")
    print("="*50)
    
    if baseline_metrics:
        print("\nBaseline (No Query Expansion):")
        for k, v in baseline_metrics.items():
            if k not in ["raw_results", "similarities"]:
                print(f"  {k}: {v:.4f}")
            
    if expanded_metrics:
        print("\nExpanded (With Query Expansion):")
        for k, v in expanded_metrics.items():
            if k not in ["raw_results", "similarities"]:
                print(f"  {k}: {v:.4f}")
            
    if baseline_metrics and expanded_metrics:
        print("\n" + "="*50)
        print("A/B COMPARISON DELTA")
        print("="*50)
        for k in baseline_metrics.keys():
            if k not in ["raw_results", "similarities"]:
                delta = expanded_metrics[k] - baseline_metrics[k]
                print(f"  {k}: Baseline={baseline_metrics[k]:.4f} | Expanded={expanded_metrics[k]:.4f} | Delta={delta:+.4f}")
                
        # Generate the HTML side-by-side report
        generate_html_report(baseline_metrics, expanded_metrics, eval_dir / "comparison_report.html")

if __name__ == "__main__":
    main()
