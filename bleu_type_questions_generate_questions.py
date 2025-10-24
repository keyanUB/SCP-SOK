import json
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import get_model, extract_json, write_json

# ----------------------------
# Config
# ----------------------------
RAW_RULES_PATH = "data/C_rules_raw.json"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_OUTPUT_PATH = OUTPUT_DIR / "title_questions.json"

GEN_MODEL_NAME = "gpt4"
NUM_RULES = 10
MAX_WORKERS = 12
MAX_RETRIES = 3
BACKOFF_BASE = 1.75

# ----------------------------
# Prompt
# ----------------------------

GEN_PROMPT = """You are creating a clean evaluation item for a secure coding rule.

Input:
- rule_title: "{rule_title}"
- rule_content: (below)

Task:
1) Convert the rule_title into a natural, self-contained **question** that asks about the rule’s guidance. Keep it direct, clear, and specific.
2) Write a **reference answer** constructed from the rule_content. Make it accurate and grounded in the content. 
   - Focus on: what the rule requires/prohibits, why it matters (risks/undefined behavior/etc.), and what a compliant approach looks like.
   - Ignore long tool/vendor lists, versions, and catalog tables.
   - Do not mention rule IDs, links, or bibliographies.
   - No length limit — answer naturally and completely.
3) Provide 3–6 **key_concepts** (short phrases) that summarize the core ideas from the answer.

Return STRICT JSON ONLY:
{{
  "question": "...",
  "reference_answer": "...",
  "key_concepts": ["...", "..."]
}}

rule_content:
{rule_content}
"""

# ----------------------------
# Helpers
# ----------------------------

def load_rules(path: str, n: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:n]

def model_generate_json(model, prompt: str) -> Dict[str, Any]:
    """Call the LLM and parse a strict JSON response with retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text = model._generate([
                {"role": "system", "content": "You respond with valid, parsable JSON only."},
                {"role": "user", "content": prompt}
            ])
            data = extract_json(text)
            if not data or not isinstance(data, dict):
                raise ValueError("Empty or invalid JSON payload")
            return data
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(BACKOFF_BASE ** (attempt - 1))

def build_item(gen_model, rule: Dict[str, Any]) -> Dict[str, Any]:
    rule_id = rule.get("rule_id", "").strip()
    rule_title = rule.get("rule_title", "").strip()
    content = rule.get("content", "").strip()

    prompt = GEN_PROMPT.format(rule_title=rule_title, rule_content=content)
    data = model_generate_json(gen_model, prompt)

    question = (data.get("question") or "").strip()
    reference = (data.get("reference_answer") or "").strip()
    key_concepts = data.get("key_concepts") or []

    # light sanitize + cap key concepts to a reasonable number
    key_concepts = [k.strip() for k in key_concepts if isinstance(k, str) and k.strip()][:8]

    return {
        "id": rule_id,
        "question": question,
        "reference": reference,
        "key_concepts": key_concepts
    }

# ----------------------------
# Main
# ----------------------------

def main():
    rules = load_rules(RAW_RULES_PATH, NUM_RULES)
    gen_model = get_model(GEN_MODEL_NAME)

    items = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(build_item, gen_model, r): i for i, r in enumerate(rules)}
        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="Building Q/A from titles"):
            try:
                items.append(fut.result())
            except Exception:
                # Skip problematic rule
                continue

    # Drop empties
    items = [it for it in items if it["id"] and it["question"] and it["reference"]]

    write_json(str(FINAL_OUTPUT_PATH), items)
    print(f"✅ Saved {len(items)} items to {FINAL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
