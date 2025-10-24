import json
import os
import re
from typing import Dict, List, Iterable
from constants import OPEN_AI_KEY, GEMINI_KEY, GEMINI_KEY, ANTHROPIC_KEY

def get_model(llm):
    if llm == 'gpt3':
        from models.GPT import OpenAIGPT
        model = OpenAIGPT(OPEN_AI_KEY, 'gpt3')
    elif llm == 'gpt4':
        from models.GPT import OpenAIGPT
        model = OpenAIGPT(OPEN_AI_KEY, 'gpt4')
    elif llm == 'gemini':
        from models.Gemini import GeminiPro
        model = GeminiPro(GEMINI_KEY)
    elif llm == 'claude':
        from models.Claude import Claude
        model = Claude(ANTHROPIC_KEY)
    else:
        import torch
        torch.cuda.empty_cache()
        if llm == 'llama3':
            from models.Llama3 import Llama3Instruct
            model = Llama3Instruct()
        elif llm == 'qwen':
            from models.Qwen import Qwen
            model = Qwen()
        elif llm == 'purpcoderl':
            from models.PurpCodeRL import PurpCodeRL
            model = PurpCodeRL()
        elif llm == 'purpcodesft':
            from models.PurpCodeSFT import PurpCodeSFT
            model = PurpCodeSFT()
        elif llm == 'codegen25':
            from models.Codegen import Codegen25Instruct
            model = Codegen25Instruct()
        elif llm == 'codellama':
            from models.CodeLlama import CodeLlama
            model = CodeLlama()
        elif llm == 'phi2':
            from models.Phi2 import Phi2
            model = Phi2()
        elif llm == 'phi2safecoder':
            from models.Phi2Safecoder import Phi2Safecoder
            model = Phi2Safecoder()
        elif llm == 'phi2sven':
            from models.Phi2Sven import Phi2Sven
            model = Phi2Sven()
        elif llm == 'mistral':
            from models.Mistral import Mistral
            model = Mistral()
        else:
            raise NotImplementedError()
    return model

def extract_json(text):
    """
    Extracts and parses a JSON object (or array) enclosed in ```json ... ``` from a text.

    Args:
        text (str): The text containing the JSON code block.

    Returns:
        dict | list | None: Parsed JSON data if found, otherwise None.
    """
    try:
        # print("The text is", text)
        # Parse and return the JSON data
        return json.loads(text)
    except:# Regex to find the JSON code block
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)

        if not match:
            return None  # No JSON block found

        json_str = match.group(1).strip()

        try:
            # Parse and return the JSON data
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing failed: {e}")
        return None

def extract_code_from_completion(completion, lang, ignore_lang=False):
    if (f'```' in completion):
        lines = completion.split('\n')
        inside_code_block = False
        current_block = []
        for line in lines:
            if (not inside_code_block and line.startswith(f'```') and (lang in line or ignore_lang)) or (inside_code_block and line.startswith(f'```')):
                if inside_code_block:
                    return '\n'.join(current_block)
                inside_code_block = not inside_code_block
                continue
            if inside_code_block:
                current_block.append(line)
    
    return ""

def remove_comments(source_code, language):
    if language in ["c", "ts", "js"]:
        # Remove single-line comments (//)
        source_code = re.sub(r'//.*', '', source_code)
        # Remove multi-line comments (/* ... */)
        source_code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)
    elif language == "py":
        # # Remove single-line comments (#)
        # source_code = re.sub(r'#.*', '', source_code)
        # # Remove multi-line comments ("""...""" or '''...''')
        # source_code = re.sub(r"('''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\")", '', source_code, flags=re.DOTALL)
        # Step 1: Remove single-line comments (#)
        source_code = re.sub(r'#.*', '', source_code)
        # Step 2: Remove standalone """...""" or '''...''' blocks without an '=' before them
        def check_assignment_before(match):
            start_index = match.start()
            preceding_text = source_code[:start_index].strip()
            # Check if '=' appears directly before the triple quotes (indicating an assignment)
            if preceding_text.endswith('='):
                return match.group(0)  # Keep as it's part of code
            return ''  # Remove as it's a standalone comment
        # Apply function to each """...""" and '''...''' block
        source_code = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', check_assignment_before, source_code, flags=re.DOTALL)
    source_code = "\n".join([line for line in source_code.split('\n') if line.strip() != ''])
    return source_code

def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\s\-_/]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_set_jaccard(a: str, b: str) -> float:
    ta = set(normalize(a).split())
    tb = set(normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)