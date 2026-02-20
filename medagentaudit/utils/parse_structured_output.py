import json
from typing import Dict
from pathlib import Path
import sys
from medagentaudit.utils.json_utils import preprocess_response_string

current_file_path = Path(__file__).resolve()
current_file_name = Path(__file__).stem
project_root = current_file_path.parents[2]
sys.path.append(str(project_root))

def parse_structured_output(response_text: str) -> Dict[str, str]:
    """
    Parse LLM response to extract structured output as a fallback.
    """
    try:
        parsed = json.loads(preprocess_response_string(response_text))
        return parsed
    except json.JSONDecodeError:
        lines = response_text.strip().split('\n')
        result = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace("'", "").replace('"', '')
                value = value.strip()
                result[key] = value

        if "explanation" not in result:
            result["explanation"] = "No structured explanation found in response"
        if "answer" not in result:
            result["answer"] = "No structured answer found in response"

        return result