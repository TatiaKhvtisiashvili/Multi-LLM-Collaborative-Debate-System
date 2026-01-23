"""
Debate Orchestrator - Main engine controlling all 5 stages
Person A Responsibility
"""
import asyncio
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path

from src.model_clients import ModelFactory, BatchProcessor, ResponseCache
from src.agent_prompts import PromptTemplates, SystemPrompts
from src.role_assigner import RoleAssigner, SelfAssessment
from src.parsing_utils import LLMOutputParser, safe_json_loads
from src.utils import setup_logging, format_timestamp

logger = logging.getLogger(__name__)


@dataclass
class DebateResult:
    """Complete result of a debate for one problem"""
    problem_id: str
    problem_text: str
    ground_truth: str
    timestamp: str

    # Stage 0 & 0.5
    role_assignments: Dict[str, str]
    self_assessments: Dict[str, Any]

    # Stage 1
    initial_solutions: Dict[str, Dict[str, Any]]

    # Stage 2
    peer_reviews: Dict[str, List[Dict[str, Any]]]  # solver_id -> list of reviews received

    # Stage 3
    refined_solutions: Dict[str, Dict[str, Any]]

    # Stage 4
    final_judgement: Dict[str, Any]

    # Metrics
    final_answer: str
    is_correct: bool
    confidence: float
    processing_time: float

    # Metadata
    model_usage: Dict[str, Any]
    errors: List[str]


class DebateOrchestrator:
    """Main orchestrator for the debate system"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.model_factory = ModelFactory()
        self.role_assigner = RoleAssigner()
        self.parser = LLMOutputParser()
        self.batch_processor = BatchProcessor(max_concurrent=3)
        self.cache = ResponseCache(self.config.get('debate_settings', {}).get('cache_dir', './cache'))

        # Initialize model clients
        self.models = {}
        self._init_models()

        # Statistics
        self.stats = {
            "problems_processed": 0,
            "total_api_calls": 0,
            "total_tokens": 0,
            "errors": 0
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML"""
        import yaml
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _init_models(self):
        """Initialize model clients"""
        model_keys = list(self.config.get('models', {}).keys())
        successful_models = 0

        for key in model_keys:
            try:
                self.models[key] = self.model_factory.create_client(key, self.config)
                logger.info(f"Initialized model: {key}")
                successful_models += 1
            except Exception as e:
                logger.error(f"Failed to initialize model {key}: {e}")

        if successful_models < 4:
            logger.error(f"Only {successful_models} models initialized, need 4")
            raise ValueError(f"Insufficient models initialized: {successful_models}/4")

    async def run_debate(self, problem: Dict[str, Any]) -> DebateResult:
        """
        Run complete debate pipeline for one problem

        Args:
            problem: Dictionary with problem data

        Returns:
            DebateResult with all stages
        """
        start_time = datetime.now()
        problem_id = problem.get('id', 'unknown')

        logger.info(f"Starting debate for problem: {problem_id}")

        result = DebateResult(
            problem_id=problem_id,
            problem_text=problem.get('problem', ''),
            ground_truth=problem.get('ground_truth_answer', ''),
            timestamp=format_timestamp(start_time),
            role_assignments={},
            self_assessments={},
            initial_solutions={},
            peer_reviews={},
            refined_solutions={},
            final_judgement={},
            final_answer='',
            is_correct=False,
            confidence=0.0,
            processing_time=0.0,
            model_usage={},
            errors=[]
        )

        try:
            # Stage 0 & 0.5: Role Assignment
            role_assignments = await self._stage_0_role_assignment(problem, result)
            result.role_assignments = role_assignments

            # Get model clients for assigned roles
            solver_clients = self._get_solver_clients(role_assignments)
            judge_client = self.models.get(role_assignments.get('judge'))

            if not solver_clients or not judge_client:
                raise ValueError("Failed to get model clients for assigned roles")

            # Stage 1: Independent Solutions
            initial_solutions = await self._stage_1_solutions(
                problem, solver_clients, role_assignments
            )
            result.initial_solutions = initial_solutions

            # Stage 2: Peer Reviews
            peer_reviews = await self._stage_2_peer_reviews(
                problem, initial_solutions, solver_clients, role_assignments
            )
            result.peer_reviews = peer_reviews

            # Stage 3: Refinement
            refined_solutions = await self._stage_3_refinement(
                problem, initial_solutions, peer_reviews, solver_clients, role_assignments
            )
            result.refined_solutions = refined_solutions

            # Stage 4: Final Judgement
            final_judgement = await self._stage_4_judgement(
                problem, initial_solutions, peer_reviews, refined_solutions, judge_client
            )
            result.final_judgement = final_judgement

            # Extract final answer
            result.final_answer = final_judgement.get('selected_final_answer', '')
            result.confidence = final_judgement.get('confidence', 0.0)

            # Check correctness
            result.is_correct = self._check_correctness(
                result.final_answer, problem.get('ground_truth_answer', '')
            )

        except Exception as e:
            error_msg = f"Error in debate for {problem_id}: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            self.stats["errors"] += 1

        # Calculate processing time
        end_time = datetime.now()
        result.processing_time = (end_time - start_time).total_seconds()

        # Update statistics
        self.stats["problems_processed"] += 1

        logger.info(f"Completed debate for {problem_id} in {result.processing_time:.2f}s")

        return result

    async def _stage_0_role_assignment(
            self, problem: Dict[str, Any], result: DebateResult
    ) -> Dict[str, str]:
        """Stage 0 & 0.5: Self-assessment and role assignment"""
        logger.info(f"Stage 0: Role assignment for {problem['id']}")

        problem_text = problem['problem']
        model_keys = list(self.models.keys())

        # Get self-assessments from all models
        assessments = {}
        tasks = []

        for model_key in model_keys:
            client = self.models[model_key]
            prompt = PromptTemplates.get_self_assessment_prompt(
                problem_text, ["Solver", "Judge"]
            )
            system_prompt = SystemPrompts.SELF_ASSESSMENT_SYSTEM_PROMPT

            tasks.append((client, prompt, system_prompt))

        # Batch process assessments
        batch_results = await self.batch_processor.process_batch(tasks)

        for (client, _, _), response in zip(tasks, batch_results):
            model_key = [k for k, v in self.models.items() if v == client][0]

            if response:
                # Parse assessment
                assessment = self.role_assigner.parse_self_assessment(
                    response.content, model_key
                )
                assessments[model_key] = assessment

                # Store in result
                result.self_assessments[model_key] = {
                    'role_preferences': assessment.role_preferences,
                    'confidence_by_role': assessment.confidence_by_role,
                    'reasoning_summary': assessment.reasoning[:200]
                }

        # Algorithmic assignment
        assignments = self.role_assigner.algorithmic_assignment(
            assessments, problem_text, model_keys
        )

        logger.info(f"Role assignments: {assignments}")
        return assignments

    def _get_solver_clients(self, assignments: Dict[str, str]) -> List[Any]:
        """Get model clients for solvers"""
        clients = []
        for i in range(1, 4):
            role = f'solver_{i}'
            model_key = assignments.get(role)
            if model_key and model_key in self.models:
                clients.append(self.models[model_key])

        return clients

    async def _stage_1_solutions(
            self,
            problem: Dict[str, Any],
            solver_clients: List[Any],
            assignments: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """Stage 1: Independent solution generation"""
        logger.info("Stage 1: Generating independent solutions")

        problem_text = problem['problem']
        solutions = {}
        tasks = []

        for i, client in enumerate(solver_clients, 1):
            solver_id = f"solver_{i}"
            prompt = PromptTemplates.get_solution_prompt(problem_text, solver_id)
            system_prompt = SystemPrompts.SOLVER_SYSTEM_PROMPT

            tasks.append((client, prompt, system_prompt, solver_id))

        # Process solutions
        for client, prompt, system_prompt, solver_id in tasks:
            try:
                # Check cache first
                cached = self.cache.get(
                    client.model_name, prompt, system_prompt
                )

                if cached:
                    response = cached
                    logger.debug(f"Using cached response for {solver_id}")
                else:
                    response = await client.generate_async(prompt, system_prompt)
                    self.cache.set(
                        client.model_name, prompt, system_prompt, response
                    )

                # Parse response
                parsed = self.parser.parse_response(response.content, "solution")
                parsed["solver_id"] = solver_id
                parsed["model_used"] = client.model_name
                parsed["raw_response"] = response.content[:1000]  # Store truncated

                solutions[solver_id] = parsed

                # Update usage stats
                self._update_usage_stats(response.usage)

            except Exception as e:
                logger.error(f"Error generating solution for {solver_id}: {e}")
                solutions[solver_id] = {
                    "solver_id": solver_id,
                    "error": str(e),
                    "parse_success": False
                }

        return solutions

    async def _stage_2_peer_reviews(
            self,
            problem: Dict[str, Any],
            solutions: Dict[str, Dict[str, Any]],
            solver_clients: List[Any],
            assignments: Dict[str, str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Stage 2: Peer review round"""
        logger.info("Stage 2: Peer reviews")

        problem_text = problem['problem']
        reviews = {solver_id: [] for solver_id in solutions.keys()}
        tasks = []

        # Create review tasks: each solver reviews the other two
        solver_ids = list(solutions.keys())

        for reviewer_idx, reviewer_id in enumerate(solver_ids):
            reviewer_client = solver_clients[reviewer_idx]

            # Get solutions to review (all except own)
            solutions_to_review = [
                (sid, sol) for sid, sol in solutions.items()
                if sid != reviewer_id
            ]

            for solution_id, solution in solutions_to_review:
                prompt = PromptTemplates.get_peer_review_prompt(
                    problem_text, solution, reviewer_id
                )
                system_prompt = SystemPrompts.REVIEWER_SYSTEM_PROMPT

                tasks.append((
                    reviewer_client, prompt, system_prompt,
                    reviewer_id, solution_id
                ))

        # Process reviews in batches
        batch_size = 2
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]

            batch_tasks = [(client, prompt, sys_prompt)
                           for client, prompt, sys_prompt, _, _ in batch]

            batch_results = await self.batch_processor.process_batch(batch_tasks)

            for (_, _, _, reviewer_id, solution_id), response in zip(batch, batch_results):
                if response:
                    # Parse review
                    parsed = self.parser.parse_response(response.content, "review")
                    parsed["reviewer_id"] = reviewer_id
                    parsed["solution_reviewed"] = solution_id

                    # Add to appropriate list
                    reviews[solution_id].append(parsed)

                    # Update usage stats
                    self._update_usage_stats(response.usage)

        return reviews

    async def _stage_3_refinement(
            self,
            problem: Dict[str, Any],
            initial_solutions: Dict[str, Dict[str, Any]],
            peer_reviews: Dict[str, List[Dict[str, Any]]],
            solver_clients: List[Any],
            assignments: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """Stage 3: Refinement based on feedback"""
        logger.info("Stage 3: Refinement")

        problem_text = problem['problem']
        refined_solutions = {}
        tasks = []

        for i, (solver_id, solution) in enumerate(initial_solutions.items()):
            client = solver_clients[i]
            reviews_for_solver = peer_reviews.get(solver_id, [])

            if not reviews_for_solver:
                logger.warning(f"No reviews for {solver_id}, skipping refinement")
                refined_solutions[solver_id] = solution
                continue

            prompt = PromptTemplates.get_refinement_prompt(
                problem_text, solution, reviews_for_solver, solver_id
            )
            system_prompt = SystemPrompts.SOLVER_SYSTEM_PROMPT

            tasks.append((client, prompt, system_prompt, solver_id))

        # Process refinements
        for client, prompt, system_prompt, solver_id in tasks:
            try:
                response = await client.generate_async(prompt, system_prompt)

                # Parse refinement
                parsed = self.parser.parse_response(response.content, "solution")
                parsed["solver_id"] = solver_id
                parsed["is_refined"] = True
                parsed["original_solution_id"] = solver_id

                refined_solutions[solver_id] = parsed

                # Update usage stats
                self._update_usage_stats(response.usage)

            except Exception as e:
                logger.error(f"Error refining solution for {solver_id}: {e}")
                refined_solutions[solver_id] = initial_solutions[solver_id]

        return refined_solutions

    async def _stage_4_judgement(
            self,
            problem: Dict[str, Any],
            initial_solutions: Dict[str, Dict[str, Any]],
            peer_reviews: Dict[str, List[Dict[str, Any]]],
            refined_solutions: Dict[str, Dict[str, Any]],
            judge_client: Any
    ) -> Dict[str, Any]:
        """Stage 4: Final judgement"""
        logger.info("Stage 4: Final judgement")

        problem_text = problem['problem']
        ground_truth = problem.get('ground_truth_answer', '')

        # Prepare all data for judge
        prompt = PromptTemplates.get_judgement_prompt(
            problem_text,
            list(initial_solutions.values()),
            self._flatten_reviews(peer_reviews),
            list(refined_solutions.values())
        )

        system_prompt = SystemPrompts.JUDGE_SYSTEM_PROMPT

        try:
            response = await judge_client.generate_async(prompt, system_prompt)

            # Parse judgement
            parsed = self.parser.parse_response(response.content, "judgement")

            # Add metadata
            parsed["judge_model"] = judge_client.model_name
            parsed["processing_timestamp"] = format_timestamp(datetime.now())

            # Update usage stats
            self._update_usage_stats(response.usage)

            return parsed

        except Exception as e:
            logger.error(f"Error in final judgement: {e}")
            return {
                "error": str(e),
                "winner": "unknown",
                "selected_final_answer": "",
                "confidence": 0.0
            }

    def _flatten_reviews(self, peer_reviews: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Flatten reviews dict to list"""
        flattened = []
        for solver_id, reviews in peer_reviews.items():
            for review in reviews:
                review["target_solver"] = solver_id
                flattened.append(review)
        return flattened

    def _check_correctness(self, final_answer: str, ground_truth: str) -> bool:
        """Check if final answer matches ground truth"""
        # Simple string comparison (could be enhanced with fuzzy matching)
        return str(final_answer).strip().lower() == str(ground_truth).strip().lower()

    def _update_usage_stats(self, usage: Dict[str, int]):
        """Update usage statistics"""
        if usage:
            self.stats["total_api_calls"] += 1
            self.stats["total_tokens"] += usage.get('total_tokens', 0)

    def save_result(self, result: DebateResult, output_dir: str = "data/results"):
        """Save debate result to file"""
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{result.problem_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)

        # Convert to dict
        result_dict = asdict(result)

        # Serialize
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved result to {filepath}")
        return filepath

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self.stats.copy()


# Helper function for batch processing
async def run_debates_batch(
        problems: List[Dict[str, Any]],
        config_path: str = "config.yaml",
        max_concurrent: int = 2
) -> List[DebateResult]:
    """
    Run debates for multiple problems with concurrency control

    Args:
        problems: List of problem dictionaries
        config_path: Path to config file
        max_concurrent: Maximum concurrent debates

    Returns:
        List of DebateResult objects
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def run_with_semaphore(problem):
        async with semaphore:
            orchestrator = DebateOrchestrator(config_path)
            result = await orchestrator.run_debate(problem)
            return result

    # Create tasks
    tasks = [run_with_semaphore(problem) for problem in problems]

    # Run with progress tracking
    from tqdm.asyncio import tqdm_asyncio

    for task in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        result = await task
        results.append(result)

    return results


if __name__ == "__main__":
    # Example usage
    import asyncio


    async def example():
        # Setup logging
        setup_logging()

        # Create orchestrator
        orchestrator = DebateOrchestrator()

        # Example problem
        problem = {
            "id": "TEST_01",
            "category": "Mathematical/Logical Reasoning",
            "problem": "What is 2 + 2?",
            "ground_truth_answer": "4",
            "ground_truth_reasoning": "Basic arithmetic"
        }

        # Run debate
        result = await orchestrator.run_debate(problem)

        # Print result
        print(f"Problem: {result.problem_id}")
        print(f"Final Answer: {result.final_answer}")
        print(f"Correct: {result.is_correct}")
        print(f"Confidence: {result.confidence}")
        print(f"Processing Time: {result.processing_time:.2f}s")

        # Save result
        orchestrator.save_result(result)


    asyncio.run(example())