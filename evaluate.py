import os
import sys
import json
import time
import datetime
import warnings
from dotenv import load_dotenv
from groq import Groq

# Ignore warnings and set environment variables
warnings.simplefilter('ignore')
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
load_dotenv()

# Import PetroChat pipeline
from petrochat import (
    load_rag_resources,
    retrieve_and_rerank,
    get_answer
)

# Output evaluation report path
EVAL_REPORT_PATH = "evaluation_report.md"

# Ground Truth Evaluation Dataset
EVAL_DATASET = [
    {
        "id": 1,
        "type": "factual",
        "query": "What are the OSHA safety requirements for tank gauging operations?",
        "expected_citations": ["[OSHA 3843, Page 6]"],
        "expected_keywords": ["alternative", "hatch", "sampling port", "pressure", "respirator", "monitor"],
        "expected_summary": "Employers should implement alternative tank gauging/sampling without opening thief hatches, retrofit dedicated sampling ports, reduce tank pressure, use appropriate respiratory protection (SAR/SCBA), use calibrated multi-gas monitors, and not permit employees to work alone."
    },
    {
        "id": 2,
        "type": "factual",
        "query": "Explain the mud weight calculation and maintenance requirements for preventing a well kick.",
        "expected_citations": ["[ABB Oil & Gas Production Handbook, Page 36]", "[API RP 54, Page 52]"],
        "expected_keywords": ["mud weight", "balance", "kill weight fluid", "location"],
        "expected_summary": "Mud weight should balance downhole pressure to prevent leakages or kicks, and adequate volumes of kill weight fluid must be maintained on location prior to flowing a well."
    },
    {
        "id": 3,
        "type": "factual",
        "query": "Describe the Process Safety Management (PSM) standard requirements for a refinery.",
        "expected_citations": ["[OSHA 3918, Page 5]", "[OSHA 3918, Page 19]"],
        "expected_keywords": ["PSI", "PHA", "Operating Procedures", "Mechanical Integrity", "MOC"],
        "expected_summary": "PSM standard requirements cover compile written Process Safety Information (PSI), Process Hazards Analysis (PHA), Operating Procedures, Mechanical Integrity (MI), and Management of Change (MOC)."
    },
    {
        "id": 4,
        "type": "factual",
        "query": "What is a blowout preventer (BOP)?",
        "expected_citations": ["[API RP 54, Page 9]"],
        "expected_keywords": ["device", "wellhead", "closed", "pipe", "wireline"],
        "expected_summary": "A blowout preventer (BOP) is a device attached to the wellhead or tree that allows the well to be closed in with or without a string of pipe or wireline in the borehole."
    },
    {
        "id": 5,
        "type": "unanswerable",
        "query": "What are the safety protocols for a blowout preventer (BOP) failure?",
        "expected_citations": [],
        "expected_keywords": [],
        "expected_summary": "I cannot answer this question based on the provided documents."
    },
    {
        "id": 6,
        "type": "unanswerable",
        "query": "What are the detailed engineering steps for designing a carbon capture system in a coal-fired power plant?",
        "expected_citations": [],
        "expected_keywords": [],
        "expected_summary": "I cannot answer this question based on the provided documents."
    }
]

def evaluate_semantic_match(client, query, generated, expected):
    """
    Queries LLaMA-3.3-70B on Groq to evaluate semantic correctness.
    """
    prompt = f"""You are an automated evaluation judge for a RAG system in the Oil & Gas domain.
Your task is to compare a Generated Answer against a Ground Truth Expected Answer for a given Query, and determine if they are semantically equivalent and factually accurate based on the expected information.

Query: {query}
Expected Answer (Ground Truth): {expected}
Generated Answer: {generated}

Rules:
1. Ignore minor stylistic or phrasing differences.
2. The Generated Answer must contain the key factual information present in the Expected Answer.
3. If the Generated Answer contains correct information but has slightly different words, mark it as PASS.
4. If the Generated Answer contradicts the Expected Answer, omits critical facts, or contains incorrect info, mark it as FAIL.
5. If the Query is unanswerable and the Generated Answer correctly states that it cannot answer based on the documents, mark it as PASS.

Provide your response in the following JSON format:
{{
  "status": "PASS" or "FAIL",
  "reason": "Brief explanation of your judgment"
}}

Do not return any other text, only the JSON block."""
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        res = json.loads(completion.choices[0].message.content.strip())
        return res.get("status", "FAIL"), res.get("reason", "No reason provided")
    except Exception as e:
        return "FAIL", f"Evaluation failed: {e}"

def main():
    print("="*60)
    print("         PETROCHAT - AUTOMATED ACCURACY EVALUATION")
    print("="*60)

    # Check API Key
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
    if not api_key:
        print("[!] ERROR: Groq API Key not found in env variables or .env file.")
        sys.exit(1)

    # Initialize Groq client
    client = Groq(api_key=api_key)

    # Load RAG Resources
    print("[+] Loading local embeddings and connecting to database...")
    try:
        db, bm25, reranker = load_rag_resources(raise_on_missing=True)
    except FileNotFoundError as e:
        print(f"[!] Database error: {e}")
        print("    Please run ingestion first using: python ingest.py")
        sys.exit(1)

    results = []
    total_cases = len(EVAL_DATASET)
    passed_cases = 0

    print(f"\n[+] Starting evaluation of {total_cases} test cases...\n")

    for case in EVAL_DATASET:
        qid = case["id"]
        q_type = case["type"]
        query = case["query"]
        expected_citations = case["expected_citations"]
        expected_summary = case["expected_summary"]
        expected_keywords = case["expected_keywords"]

        print(f"--- Running Case {qid}/{total_cases} ({q_type}): '{query}' ---")

        # Step 1: Retrieve and Rerank
        retrieved_results = retrieve_and_rerank(query, db, bm25, reranker, top_n=3)

        # Step 2: Generate Answer
        start_time = time.time()
        answer = get_answer(client, query, retrieved_results, [])
        latency = time.time() - start_time

        # Step 3: Grade Citations
        citation_pass = True
        missing_citations = []
        for citation in expected_citations:
            if citation not in answer:
                citation_pass = False
                missing_citations.append(citation)

        # Step 4: Grade Semantic Correctness via LLM Judge
        if q_type == "unanswerable":
            # Direct check for guardrail rejection
            expected_guardrail = "I cannot answer this question based on the provided documents"
            semantic_pass = expected_guardrail.lower() in answer.lower()
            judge_reason = "Correctly rejected out-of-context query." if semantic_pass else "Failed to trigger hallucination guardrails."
            status = "PASS" if semantic_pass else "FAIL"
        else:
            status, judge_reason = evaluate_semantic_match(client, query, answer, expected_summary)
            semantic_pass = (status == "PASS")

        # Case passes if both citations and semantic checks succeed
        case_passed = citation_pass and semantic_pass
        if case_passed:
            passed_cases += 1

        # Check keyword overlaps
        matched_keywords = [kw for kw in expected_keywords if kw.lower() in answer.lower()]
        keyword_overlap_pct = (len(matched_keywords) / len(expected_keywords) * 100) if expected_keywords else 100.0

        print(f"    -> Graded: {status} | Latency: {latency:.2f}s")
        if not citation_pass:
            print(f"    -> Missing Citations: {missing_citations}")

        results.append({
            "id": qid,
            "type": q_type,
            "query": query,
            "answer": answer,
            "expected_summary": expected_summary,
            "expected_citations": expected_citations,
            "citation_pass": citation_pass,
            "missing_citations": missing_citations,
            "semantic_pass": semantic_pass,
            "judge_reason": judge_reason,
            "keyword_overlap_pct": keyword_overlap_pct,
            "latency": latency,
            "passed": case_passed
        })
        print("-" * 50)

    # Compute stats
    overall_accuracy = (passed_cases / total_cases) * 100
    avg_latency = sum(r["latency"] for r in results) / total_cases
    factual_cases = [r for r in results if r["type"] == "factual"]
    unanswerable_cases = [r for r in results if r["type"] == "unanswerable"]

    factual_pass = sum(1 for r in factual_cases if r["passed"])
    factual_accuracy = (factual_pass / len(factual_cases) * 100) if factual_cases else 100.0

    guardrail_pass = sum(1 for r in unanswerable_cases if r["passed"])
    guardrail_accuracy = (guardrail_pass / len(unanswerable_cases) * 100) if unanswerable_cases else 100.0

    print(f"\n[+] Evaluation Complete!")
    print(f"    -> Overall Accuracy: {overall_accuracy:.1f}% ({passed_cases}/{total_cases})")
    print(f"    -> Factual Correctness: {factual_accuracy:.1f}%")
    print(f"    -> Guardrail Accuracy: {guardrail_accuracy:.1f}%")
    print(f"    -> Average Latency: {avg_latency:.2f}s")

    # Generate Markdown Report
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = []
    report.append(f"# PetroChat - RAG Evaluation Report ({timestamp})\n")
    report.append("This report details the automated accuracy evaluation of the PetroChat RAG pipeline using Ground Truth factual and unanswerable questions.\n")
    
    report.append("## 📈 Performance Summary\n")
    report.append(f"- **Overall Pass Rate**: `{overall_accuracy:.1f}%` ({passed_cases}/{total_cases} cases)")
    report.append(f"- **Factual Query Accuracy**: `{factual_accuracy:.1f}%` ({factual_pass}/{len(factual_cases)} cases)")
    report.append(f"- **Guardrail Rejection Rate**: `{guardrail_accuracy:.1f}%` ({guardrail_pass}/{len(unanswerable_cases)} cases)")
    report.append(f"- **Average LLM Generation Latency**: `{avg_latency:.2f} seconds` \n")
    
    report.append("## 🔍 Detailed Test Case Results\n")
    
    for r in results:
        status_emoji = "✅ PASS" if r["passed"] else "❌ FAIL"
        report.append(f"### Case {r['id']} ({r['type']}): {r['query']}\n")
        report.append(f"- **Result Status**: {status_emoji}")
        report.append(f"- **Latency**: `{r['latency']:.2f}s` | **Keyword Overlap**: `{r['keyword_overlap_pct']:.1f}%` ")
        report.append(f"- **Citation Check**: `{'Pass' if r['citation_pass'] else 'Fail'}` (Expected: `{r['expected_citations']}`)")
        if r["missing_citations"]:
            report.append(f"  - *Missing Citations*: `{r['missing_citations']}`")
        report.append(f"- **LLM Judge Evaluation**: `{'Pass' if r['semantic_pass'] else 'Fail'}`")
        report.append(f"  - *Reasoning*: {r['judge_reason']}\n")
        report.append("#### Generated Answer:")
        report.append(f"> {r['answer']}\n")
        report.append("--- \n")

    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"\n[+] Detailed evaluation report saved to '{EVAL_REPORT_PATH}'.")

if __name__ == "__main__":
    main()
