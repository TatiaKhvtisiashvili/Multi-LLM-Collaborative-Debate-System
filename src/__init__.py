"""
Multi-LLM Collaborative Debate System - Main Package
"""

from src.debate_orchestrator import DebateOrchestrator, DebateResult, run_debates_batch
from src.model_clients import (
    ModelClient,
    GroqClient,
    MistralClient,
    ModelFactory,
    BatchProcessor,
    ResponseCache,
    ModelResponse
)
from src.agent_prompts import PromptTemplates, SystemPrompts
# Try to import PromptSelector, but don't fail if it doesn't exist
try:
    from src.agent_prompts import PromptSelector
except ImportError:
    PromptSelector = None  # Set to None if not available
from src.role_assigner import RoleAssigner, SelfAssessment, JudgeVote
from src.dataset_manager import DatasetManager, Problem
from src.evaluation_metrics import (
    MetricsCalculator,
    SystemMetrics,
    ProblemMetrics,
    BaselineExperiment
)
from src.parsing_utils import LLMOutputParser, ParseStatistics, extract_final_answer, safe_json_loads
from src.utils import (
    setup_logging,
    format_timestamp,
    load_json_file,
    save_json_file,
    calculate_elapsed_time,
    clean_string,
    safe_divide,
    get_project_root,
    ensure_directory,
    Timer,
    batch_process,
    get_env_var,
    parse_bool
)

__version__ = "1.0.0"
__author__ = "Multi-LLM Debate System Team"
__description__ = "A collaborative debate system where multiple LLMs solve problems through structured critique and refinement"

# Export main classes for easy imports
__all__ = [
    # Main orchestrator
    "DebateOrchestrator",
    "DebateResult",
    "run_debates_batch",

    # Model clients
    "ModelClient",
    "GroqClient",
    "MistralClient",
    "ModelFactory",
    "BatchProcessor",
    "ResponseCache",
    "ModelResponse",

    # Prompts and templates
    "PromptTemplates",
    "SystemPrompts",
    "PromptSelector",  # This might be None

    # Role assignment
    "RoleAssigner",
    "SelfAssessment",
    "JudgeVote",

    # Dataset management
    "DatasetManager",
    "Problem",

    # Evaluation
    "MetricsCalculator",
    "SystemMetrics",
    "ProblemMetrics",
    "BaselineExperiment",

    # Parsing utilities
    "LLMOutputParser",
    "ParseStatistics",
    "extract_final_answer",
    "safe_json_loads",

    # General utilities
    "setup_logging",
    "format_timestamp",
    "load_json_file",
    "save_json_file",
    "calculate_elapsed_time",
    "clean_string",
    "safe_divide",
    "get_project_root",
    "ensure_directory",
    "Timer",
    "batch_process",
    "get_env_var",
    "parse_bool",

    # Constants
    "__version__",
    "__author__",
    "__description__"
]
