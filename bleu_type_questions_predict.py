import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from tqdm import tqdm

from utils import get_model, write_json

# ----------------------------
# Config
# ----------------------------
INPUT_PATH = Path("outputs/title_questions.json")        # from the previous step
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "candidates_from_titles.json"

CANDIDATE_MODEL_NAME = "gpt4"    # Model under test
MAX_WORKERS = 16
MAX_RETRIES = 3
BACKOFF_BASE = 1.75

# ----------------------------
# Prompt (no length restriction)
# ----------------------------
SYS_PROMPT = (
    "Answer clearly, accurately, and concisely in a couple of sentences. At most 20 sentences."
)

USER_PROMPT_TEMPLATE = """Question:
{question}
"""

# ----------------------------
# Helpers
# ----------------------------

def read_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def model_generate_text(model, sys_prompt: str, user_prompt: str) -> str:
    """Call model with retries; return stripped text or empty string on final failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = model._generate([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ])
            return (resp or "").strip()
        except Exception:
            if attempt == MAX_RETRIES:
                return ""
            time.sleep(BACKOFF_BASE ** (attempt - 1))

def build_candidate(model, item: Dict[str, Any]) -> Dict[str, Any]:
    """Generate candidate answer for a single item."""
    question = item.get("question", "").strip()
    user_prompt = USER_PROMPT_TEMPLATE.format(question=question)
    candidate = model_generate_text(model, SYS_PROMPT, user_prompt)

    return {
        "id": item.get("id", ""),
        "question": question,
        "reference": item.get("reference", ""),
        "candidate": candidate,
        "key_concepts": item.get("key_concepts", []),
    }

# ----------------------------
# Main
# ----------------------------

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    items = read_json(INPUT_PATH)
    if not items:
        write_json(str(OUTPUT_PATH), [])
        print("No input items found. Wrote empty output.")
        return

    model = get_model(CANDIDATE_MODEL_NAME)

    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_idx = {
            pool.submit(build_candidate, model, item): i
            for i, item in enumerate(items)
        }
        for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Generating candidates"):
            idx = future_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                # fallback empty candidate while preserving order
                it = items[idx]
                results[idx] = {
                    "id": it.get("id", ""),
                    "question": it.get("question", ""),
                    "reference": it.get("reference", ""),
                    "candidate": "",
                    "key_concepts": it.get("key_concepts", []),
                }

    # Filter any totally empty rows (shouldn’t happen often)
    results = [
        r for r in results
        if r and r.get("id") and r.get("question") and r.get("reference") is not None
    ]

    write_json(str(OUTPUT_PATH), results)
    print(f"✅ Saved {len(results)} predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
