"""
Parsing Utilities - Cleans and validates LLM outputs
Person B Responsibility
"""
import json
import re
import ast
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class LLMOutputParser:
    """Handles parsing and validation of LLM outputs"""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.error_count = 0
        self.fix_count = 0

    def parse_response(self, response_text: str, expected_type: str = "json") -> Dict[str, Any]:
        """
        Parse LLM response with robust error handling

        Args:
            response_text: Raw LLM output
            expected_type: "json", "solution", "review", "judgement"

        Returns:
            Parsed dictionary
        """
        if not response_text or response_text.strip() == "":
            return self._create_error_response("Empty response", expected_type)

        # Clean the response
        cleaned_text = self._clean_response(response_text)

        try:
            if expected_type == "json" or expected_type in ["solution", "review", "judgement"]:
                return self._parse_json_response(cleaned_text, expected_type)
            else:
                # Fallback to text extraction
                return {"content": cleaned_text, "parse_success": True}
        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")
            self.error_count += 1

            # Try recovery strategies
            recovered = self._recover_parse(cleaned_text, expected_type)
            if recovered:
                self.fix_count += 1
                return recovered

            return self._create_error_response(str(e), expected_type)

    def _clean_response(self, text: str) -> str:
        """Clean common LLM response artifacts"""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        # Remove thought/chain-of-thought prefixes
        text = re.sub(r'^(Thought:|Reasoning:|Step \d+:)\s*', '', text, flags=re.MULTILINE)

        # Remove excessive whitespace but preserve structure
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Fix common LLM quirks
        text = text.replace('\\"', '"')  # Unescape quotes
        text = text.replace('\\n', '\n')  # Unescape newlines

        return text.strip()

    def _parse_json_response(self, text: str, expected_type: str) -> Dict[str, Any]:
        """Parse JSON response with validation"""
        # First try direct JSON parse
        try:
            data = json.loads(text)
            return self._validate_and_fix_structure(data, expected_type)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from text
        json_match = self._extract_json_from_text(text)
        if json_match:
            try:
                data = json.loads(json_match)
                return self._validate_and_fix_structure(data, expected_type)
            except json.JSONDecodeError:
                pass

        # Try Python dict literal (LLMs sometimes output this)
        try:
            data = ast.literal_eval(text)
            if isinstance(data, dict):
                return self._validate_and_fix_structure(data, expected_type)
        except (SyntaxError, ValueError):
            pass

        # Last resort: manual extraction
        return self._manual_extraction(text, expected_type)

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON-like structure from text"""
        # Find outermost braces
        brace_stack = []
        start_idx = -1

        for i, char in enumerate(text):
            if char == '{':
                if not brace_stack:
                    start_idx = i
                brace_stack.append('{')
            elif char == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start_idx != -1:
                        # Found complete JSON
                        return text[start_idx:i + 1]

        # If we get here, try regex as fallback
        json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}'  # Match nested objects
        match = re.search(json_pattern, text)
        if match:
            return match.group(0)

        return None

    def _validate_and_fix_structure(self, data: Dict[str, Any], expected_type: str) -> Dict[str, Any]:
        """Validate and fix common structural issues"""
        if not isinstance(data, dict):
            data = {"content": str(data)}

        # Add metadata
        data["_metadata"] = {
            "parsed_at": datetime.now().isoformat(),
            "expected_type": expected_type,
            "parse_success": True,
            "was_fixed": False
        }

        # Type-specific validation and fixing
        if expected_type == "solution":
            data = self._fix_solution_structure(data)
        elif expected_type == "review":
            data = self._fix_review_structure(data)
        elif expected_type == "judgement":
            data = self._fix_judgement_structure(data)

        return data

    def _fix_solution_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix common issues in solution structure"""
        # Ensure required fields
        if "solver_id" not in data:
            data["solver_id"] = "unknown"

        if "solution_steps" not in data:
            # Try to extract steps from text
            if "solution" in data and isinstance(data["solution"], str):
                steps = self._extract_steps_from_text(data["solution"])
                data["solution_steps"] = steps
            else:
                data["solution_steps"] = [{"step": 1, "description": "No steps provided"}]

        if "final_answer" not in data:
            # Try to extract answer
            if "answer" in data:
                data["final_answer"] = str(data["answer"])
            else:
                data["final_answer"] = "No answer provided"

        if "confidence" not in data:
            data["confidence"] = 0.5

        # Ensure solution_steps is a list of dicts
        if isinstance(data["solution_steps"], str):
            data["solution_steps"] = [{"step": 1, "description": data["solution_steps"]}]
        elif isinstance(data["solution_steps"], list):
            for i, step in enumerate(data["solution_steps"]):
                if isinstance(step, str):
                    data["solution_steps"][i] = {"step": i + 1, "description": step}

        data["_metadata"]["was_fixed"] = True
        return data

    def _fix_review_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix common issues in review structure"""
        # Ensure evaluation field
        if "evaluation" not in data:
            data["evaluation"] = {}

        eval_field = data["evaluation"]

        # Ensure all subfields exist
        for field in ["strengths", "weaknesses", "errors", "suggested_changes"]:
            if field not in eval_field:
                eval_field[field] = []

        # Fix errors structure
        if isinstance(eval_field["errors"], list):
            for i, error in enumerate(eval_field["errors"]):
                if isinstance(error, str):
                    eval_field["errors"][i] = {
                        "location": "unknown",
                        "error_type": "general",
                        "description": error,
                        "severity": "moderate"
                    }

        if "overall_assessment" not in data:
            # Infer from errors
            if eval_field["errors"]:
                data["overall_assessment"] = "flawed"
            else:
                data["overall_assessment"] = "mostly_correct"

        data["_metadata"]["was_fixed"] = True
        return data

    def _fix_judgement_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix common issues in judgement structure"""
        # Ensure judgement field
        if "judgement" not in data:
            data["judgement"] = {}

        judgement = data["judgement"]

        # Ensure winner field
        if "winner" not in judgement:
            # Try to infer
            if "ranking" in judgement and isinstance(judgement["ranking"], list):
                if judgement["ranking"]:
                    judgement["winner"] = judgement["ranking"][0].get("solver", "unknown")
            else:
                judgement["winner"] = "unknown"

        # Ensure ranking field
        if "ranking" not in judgement:
            judgement["ranking"] = []

        # Ensure selected_final_answer
        if "selected_final_answer" not in data:
            data["selected_final_answer"] = "No answer selected"

        data["_metadata"]["was_fixed"] = True
        return data

    def _extract_steps_from_text(self, text: str) -> List[Dict[str, str]]:
        """Extract steps from free-text solution"""
        steps = []

        # Look for numbered steps
        step_pattern = r'(?:Step\s+(\d+)[:.]?\s*|(\d+)\.\s*|•\s*)(.+?)(?=(?:Step\s+\d+|$\n\d+\.|\n•))'
        matches = re.findall(step_pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            step_num = match[0] or match[1] or str(len(steps) + 1)
            description = match[2].strip()
            steps.append({
                "step": int(step_num) if step_num.isdigit() else len(steps) + 1,
                "description": description
            })

        if not steps:
            # Fallback: split by sentences or lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            for i, line in enumerate(lines[:10]):  # Limit to 10 steps
                steps.append({
                    "step": i + 1,
                    "description": line[:500]  # Limit description length
                })

        return steps

    def _manual_extraction(self, text: str, expected_type: str) -> Dict[str, Any]:
        """Manual extraction when all else fails"""
        result = {
            "raw_content": text[:1000],  # Truncate for storage
            "parse_success": False,
            "_metadata": {
                "parsed_at": datetime.now().isoformat(),
                "expected_type": expected_type,
                "parse_success": False,
                "error": "Failed to parse, using manual extraction"
            }
        }

        # Try to extract key information based on type
        if expected_type == "solution":
            # Look for final answer patterns
            answer_patterns = [
                r'[Ff]inal [Aa]nswer[:]?\s*(.+)',
                r'[Aa]nswer[:]?\s*(.+)',
                r'[Ss]olution[:]?\s*(.+)'
            ]

            for pattern in answer_patterns:
                match = re.search(pattern, text)
                if match:
                    result["extracted_answer"] = match.group(1).strip()
                    break

        elif expected_type == "review":
            # Look for assessment words
            assessment_words = ["correct", "incorrect", "flawed", "good", "bad", "error", "mistake"]
            found = []
            for word in assessment_words:
                if re.search(rf'\b{word}\b', text, re.IGNORECASE):
                    found.append(word)

            if found:
                result["extracted_assessment"] = ", ".join(found[:3])

        return result

    def _create_error_response(self, error_msg: str, expected_type: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "error": error_msg,
            "parse_success": False,
            "_metadata": {
                "parsed_at": datetime.now().isoformat(),
                "expected_type": expected_type,
                "parse_success": False,
                "error_message": error_msg
            }
        }

    def _recover_parse(self, text: str, expected_type: str) -> Optional[Dict[str, Any]]:
        """Attempt to recover from parse failure"""
        recovery_strategies = [
            self._try_fix_trailing_commas,
            self._try_fix_unquoted_keys,
            self._try_fix_single_quotes,
            self._try_extract_with_llm_fallback  # Would use a simple LLM call in production
        ]

        for strategy in recovery_strategies:
            try:
                result = strategy(text, expected_type)
                if result:
                    return result
            except:
                continue

        return None

    def _try_fix_trailing_commas(self, text: str, expected_type: str) -> Optional[Dict[str, Any]]:
        """Fix trailing commas in JSON"""
        fixed = re.sub(r',\s*}', '}', text)
        fixed = re.sub(r',\s*]', ']', fixed)

        if fixed != text:
            try:
                return json.loads(fixed)
            except:
                pass

        return None

    def _try_fix_unquoted_keys(self, text: str, expected_type: str) -> Optional[Dict[str, Any]]:
        """Fix unquoted keys in JSON"""

        # Simple pattern: word followed by colon (not in quotes)
        def quote_key(match):
            key = match.group(1)
            if not (key.startswith('"') or key.startswith("'")):
                return f'"{key}":'
            return match.group(0)

        fixed = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:', quote_key, text)

        if fixed != text:
            try:
                return json.loads(fixed)
            except:
                pass

        return None

    def _try_fix_single_quotes(self, text: str, expected_type: str) -> Optional[Dict[str, Any]]:
        """Convert single quotes to double quotes"""
        # Simple conversion (not perfect but works for many cases)
        fixed = text.replace("'", '"')

        if fixed != text:
            try:
                return json.loads(fixed)
            except:
                pass

        return None

    def _try_extract_with_llm_fallback(self, text: str, expected_type: str) -> Optional[Dict[str, Any]]:
        """Use a simple LLM to fix JSON (simulated here)"""
        # In production, this would call a small/fast LLM to fix the JSON
        # For now, return None to indicate failure
        return None


# Statistics tracker
class ParseStatistics:
    """Track parsing statistics"""

    def __init__(self):
        self.total_parses = 0
        self.successful_parses = 0
        self.fixed_parses = 0
        self.failed_parses = 0
        self.errors_by_type = {}

    def log_parse(self, success: bool, was_fixed: bool = False, error_type: str = None):
        self.total_parses += 1

        if success:
            self.successful_parses += 1
            if was_fixed:
                self.fixed_parses += 1
        else:
            self.failed_parses += 1
            if error_type:
                self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_parses": self.total_parses,
            "success_rate": self.successful_parses / self.total_parses if self.total_parses > 0 else 0,
            "fix_rate": self.fixed_parses / self.total_parses if self.total_parses > 0 else 0,
            "failure_rate": self.failed_parses / self.total_parses if self.total_parses > 0 else 0,
            "common_errors": dict(sorted(self.errors_by_type.items(), key=lambda x: x[1], reverse=True)[:5])
        }


# Utility function for common parsing tasks
def safe_json_loads(text: str, default=None):
    """Safely parse JSON with fallback"""
    parser = LLMOutputParser(strict_mode=False)
    result = parser.parse_response(text, "json")

    if result.get("parse_success", False):
        # Remove metadata for clean output
        result.pop("_metadata", None)
        result.pop("parse_success", None)
        return result
    else:
        return default if default is not None else {"error": "Failed to parse JSON", "raw_text": text[:500]}


def extract_final_answer(text: str) -> str:
    """Extract final answer from text"""
    patterns = [
        r'[Ff]inal [Aa]nswer\s*[:=]?\s*["\']?([^"\'\n]+)["\']?',
        r'[Aa]nswer\s*[:=]?\s*["\']?([^"\'\n]+)["\']?',
        r'[Tt]he [Aa]nswer [Ii]s\s*[:=]?\s*["\']?([^"\'\n]+)["\']?'
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # Fallback: last non-empty line
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        return lines[-1][:200]  # Truncate

    return "No answer found"