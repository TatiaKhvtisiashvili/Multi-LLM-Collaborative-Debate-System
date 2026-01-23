"""
Multi-LLM Collaborative Debate System - Main Package

This package implements a debate system where multiple LLMs collaboratively solve problems
through independent solution generation, peer review, refinement, and final judgement.

Main Components:
1. DebateOrchestrator: Main engine controlling all debate stages
2. ModelClients: Interface to various LLM APIs (Groq, Mistral, etc.)
3. AgentPrompts: System prompts and templates for all roles
4. RoleAssigner: Assigns roles to models based on self-assessment
5. DatasetManager: Handles problem dataset creation and management
6. EvaluationMetrics: Calculates quantitative metrics for system performance
7. ParsingUtils: Cleans and validates LLM outputs
8. Utils: General utility functions
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
from src.agent_prompts import PromptTemplates, SystemPrompts, PromptSelector
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
    "PromptSelector",

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


def get_system_info() -> dict:
    """
    Get information about the debate system

    Returns:
        Dictionary with system information
    """
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "components": {
            "orchestrator": "DebateOrchestrator",
            "models": ["GroqClient", "MistralClient"],
            "prompts": "PromptTemplates",
            "roles": "RoleAssigner",
            "dataset": "DatasetManager",
            "evaluation": "MetricsCalculator",
            "parsing": "LLMOutputParser"
        }
    }


def validate_environment() -> bool:
    """
    Validate that required environment variables are set

    Returns:
        True if all required environment variables are set
    """
    required_vars = ["GROQ_API_KEY", "MISTRAL_API_KEY"]
    missing = []

    for var in required_vars:
        if not get_env_var(var):
            missing.append(var)

    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("Please set these variables in your environment:")
        for var in missing:
            print(f"  export {var}=your_api_key_here")
        return False

    print("✅ All required environment variables are set")
    return True


def quick_start(problem_text: str = None) -> DebateResult:
    """
    Quick start function for testing the debate system

    Args:
        problem_text: Optional problem text to test

    Returns:
        DebateResult object
    """
    import asyncio
    from datetime import datetime

    if not problem_text:
        problem_text = "What is 12 × 13?"

    problem = {
        "id": "QUICK_START_TEST",
        "category": "Mathematical/Logical Reasoning",
        "problem": problem_text,
        "ground_truth_answer": "156" if "12 × 13" in problem_text else "unknown",
        "ground_truth_reasoning": "Test problem for quick start"
    }

    print(f"🚀 Starting quick debate for: {problem_text[:50]}...")

    try:
        orchestrator = DebateOrchestrator()
        result = asyncio.run(orchestrator.run_debate(problem))

        print(f"✅ Debate completed in {result.processing_time:.2f}s")
        print(f"📊 Final answer: {result.final_answer}")
        print(f"🎯 Correct: {result.is_correct}")

        return result
    except Exception as e:
        print(f"❌ Error in quick start: {e}")
        raise


# Example usage when module is run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Multi-LLM Collaborative Debate System")
    print(f"Version: {__version__}")
    print("=" * 60)

    # Show system info
    info = get_system_info()
    print(f"Author: {info['author']}")
    print(f"Description: {info['description']}")
    print()

    # Validate environment
    print("🔍 Validating environment...")
    if validate_environment():
        print("\n✅ System is ready to use!")
        print("\nUsage example:")
        print("  from src import DebateOrchestrator, DatasetManager")
        print("  import asyncio")
        print()
        print("  # Load problems")
        print("  manager = DatasetManager('data/problems.json')")
        print("  problems = manager.load_dataset()")
        print()
        print("  # Run debate")
        print("  orchestrator = DebateOrchestrator()")
        print("  result = asyncio.run(orchestrator.run_debate(problems[0]))")
        print()
        print("  # Evaluate results")
        print("  calculator = MetricsCalculator()")
        print("  calculator.load_results()")
        print("  metrics = calculator.calculate_system_metrics()")
        print(f"  print(f'Accuracy: {metrics.overall_accuracy:.2%}')")
    else:
        print("\nPlease set the required environment variables first.")

    print("\n" + "=" * 60)