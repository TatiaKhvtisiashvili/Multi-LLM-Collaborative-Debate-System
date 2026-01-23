"""
Utility functions for the debate system
"""
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
import json


def setup_logging(log_file: str = None, level: str = "INFO"):
    """Configure logging for the application"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def format_timestamp(dt: datetime = None) -> str:
    """Format datetime to ISO string"""
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON file with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")


def save_json_file(data: Dict[str, Any], filepath: str, indent: int = 2):
    """Save data to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def calculate_elapsed_time(start_time: datetime) -> str:
    """Calculate and format elapsed time"""
    elapsed = datetime.now() - start_time
    total_seconds = int(elapsed.total_seconds())

    if total_seconds < 60:
        return f"{total_seconds}s"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def clean_string(text: str, max_length: int = 500) -> str:
    """Clean string by removing extra whitespace and truncating"""
    if not text:
        return ""

    # Remove extra whitespace
    cleaned = ' '.join(text.split())

    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."

    return cleaned


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with zero denominator handling"""
    if denominator == 0:
        return default
    return numerator / denominator


def get_project_root() -> Path:
    """Get the project root directory"""
    current_file = Path(__file__).resolve()

    # Navigate up until we find requirements.txt or .git
    for parent in [current_file] + list(current_file.parents):
        if (parent / 'requirements.txt').exists() or (parent / '.git').exists():
            return parent

    # Fallback to current file's parent
    return current_file.parent.parent


def ensure_directory(path: str):
    """Ensure directory exists, create if it doesn't"""
    Path(path).mkdir(parents=True, exist_ok=True)


class Timer:
    """Context manager for timing code blocks"""

    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, *args):
        self.end = datetime.now()
        self.elapsed = self.end - self.start

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds"""
        return self.elapsed.total_seconds()

    def get_formatted(self) -> str:
        """Get formatted elapsed time"""
        return calculate_elapsed_time(self.start)


def batch_process(items: List[Any], batch_size: int):
    """Yield batches of items"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def is_valid_email(email: str) -> bool:
    """Simple email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


# Environment variable helpers
def get_env_var(key: str, default: str = None, required: bool = False) -> str:
    """Get environment variable with validation"""
    value = os.getenv(key, default)

    if required and value is None:
        raise ValueError(f"Environment variable {key} is required but not set")

    return value


def parse_bool(value: str) -> bool:
    """Parse string to boolean"""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() in ('true', 'yes', '1', 't', 'y')

    return bool(value)


if __name__ == "__main__":
    # Test the utilities
    logger = setup_logging()
    logger.info("Testing utilities...")

    # Test Timer
    with Timer() as timer:
        import time

        time.sleep(0.1)

    print(f"Elapsed time: {timer.get_formatted()}")

    # Test string cleaning
    test_text = "  Hello   world!  " * 100
    cleaned = clean_string(test_text, 50)
    print(f"Cleaned string: {cleaned}")

    logger.info("Utilities test complete")