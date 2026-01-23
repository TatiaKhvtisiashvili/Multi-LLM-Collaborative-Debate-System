"""
Evaluation Metrics - Calculates quantitative metrics for system performance
Person A Responsibility
"""
import json
import os
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from pathlib import Path


@dataclass
class ProblemMetrics:
    """Metrics for a single problem"""
    problem_id: str
    category: str
    system_correct: bool
    system_answer: str
    ground_truth: str
    solver_answers: List[str]
    refined_answers: List[str]
    final_judgement: Dict[str, Any]
    solver_consensus: bool  # All solvers gave same initial answer
    refinement_changed: bool  # Any solver changed answer after review
    judge_picked_correct: bool  # When solvers disagreed, did judge pick right?
    processing_time: float


@dataclass
class SystemMetrics:
    """Aggregated system performance metrics"""
    overall_accuracy: float
    improvement_rate: float
    consensus_rate: float
    judge_accuracy: float
    avg_processing_time: float

    # Category-wise breakdown
    category_accuracy: Dict[str, float]
    category_consensus: Dict[str, float]

    # Confidence analysis
    avg_confidence_when_correct: float
    avg_confidence_when_wrong: float
    confidence_correlation: float

    # Solver performance
    solver_individual_accuracy: Dict[str, float]
    solver_contribution_to_wins: Dict[str, int]

    # Error analysis
    common_error_types: List[Tuple[str, int]]
    problems_with_contradictions: List[str]


class MetricsCalculator:
    """Calculates evaluation metrics from debate results"""

    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.results: List[Dict[str, Any]] = []
        self.problems_metrics: List[ProblemMetrics] = []
        self.system_metrics: SystemMetrics = None

    def load_results(self, pattern: str = "*.json"):
        """Load all result files from directory"""
        result_files = list(self.results_dir.glob(pattern))

        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    self.results.append(result)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        print(f"Loaded {len(self.results)} result files")
        return self

    def calculate_problem_metrics(self) -> List[ProblemMetrics]:
        """Calculate metrics for each individual problem"""
        self.problems_metrics = []

        for result in self.results:
            try:
                # Extract solver answers
                initial_answers = []
                refined_answers = []

                for solver_id in ['solver_1', 'solver_2', 'solver_3']:
                    if solver_id in result.get('initial_solutions', {}):
                        sol = result['initial_solutions'][solver_id]
                        if sol.get('parse_success', True):
                            answer = sol.get('final_answer', '')
                            if answer:
                                initial_answers.append(str(answer).strip().lower())

                    if solver_id in result.get('refined_solutions', {}):
                        sol = result['refined_solutions'][solver_id]
                        if sol.get('parse_success', True):
                            answer = sol.get('final_answer', '')
                            if answer:
                                refined_answers.append(str(answer).strip().lower())

                # Check consensus (all initial answers same)
                if len(initial_answers) >= 2:
                    consensus = all(a == initial_answers[0] for a in initial_answers)
                else:
                    consensus = False

                # Check if refinement changed any answers
                refinement_changed = False
                if initial_answers and refined_answers and len(initial_answers) == len(refined_answers):
                    for init, refined in zip(initial_answers, refined_answers):
                        if init != refined:
                            refinement_changed = True
                            break

                # System answer from final judgement
                system_answer = result.get('final_answer', '').strip().lower()
                ground_truth = result.get('ground_truth', '').strip().lower()
                system_correct = system_answer == ground_truth

                # Judge accuracy (when solvers disagreed, did judge pick correct one?)
                judge_picked_correct = False
                if not consensus and initial_answers:
                    # Check if correct answer was among solver answers
                    correct_in_solutions = ground_truth in initial_answers
                    if correct_in_solutions:
                        # Check if judge picked a solver with correct answer
                        winner = result.get('final_judgement', {}).get('winner', '')
                        if winner in result.get('initial_solutions', {}):
                            winner_sol = result['initial_solutions'][winner]
                            winner_answer = str(winner_sol.get('final_answer', '')).strip().lower()
                            judge_picked_correct = winner_answer == ground_truth

                metrics = ProblemMetrics(
                    problem_id=result.get('problem_id', 'unknown'),
                    category=result.get('problem_text', '')[:50],  # Simplified
                    system_correct=system_correct,
                    system_answer=system_answer,
                    ground_truth=ground_truth,
                    solver_answers=initial_answers,
                    refined_answers=refined_answers,
                    final_judgement=result.get('final_judgement', {}),
                    solver_consensus=consensus,
                    refinement_changed=refinement_changed,
                    judge_picked_correct=judge_picked_correct,
                    processing_time=result.get('processing_time', 0)
                )

                self.problems_metrics.append(metrics)

            except Exception as e:
                print(f"Error calculating metrics for {result.get('problem_id', 'unknown')}: {e}")

        return self.problems_metrics

    def calculate_system_metrics(self) -> SystemMetrics:
        """Calculate overall system metrics"""
        if not self.problems_metrics:
            self.calculate_problem_metrics()

        n_problems = len(self.problems_metrics)
        if n_problems == 0:
            return SystemMetrics(
                overall_accuracy=0,
                improvement_rate=0,
                consensus_rate=0,
                judge_accuracy=0,
                avg_processing_time=0,
                category_accuracy={},
                category_consensus={},
                avg_confidence_when_correct=0,
                avg_confidence_when_wrong=0,
                confidence_correlation=0,
                solver_individual_accuracy={},
                solver_contribution_to_wins={},
                common_error_types=[],
                problems_with_contradictions=[]
            )

        # Basic metrics
        n_correct = sum(1 for m in self.problems_metrics if m.system_correct)
        n_refinement_changed = sum(1 for m in self.problems_metrics if m.refinement_changed)
        n_consensus = sum(1 for m in self.problems_metrics if m.solver_consensus)

        # Judge accuracy (only consider problems without consensus)
        problems_without_consensus = [m for m in self.problems_metrics if not m.solver_consensus]
        n_judge_correct = sum(1 for m in problems_without_consensus if m.judge_picked_correct)

        overall_accuracy = n_correct / n_problems
        improvement_rate = n_refinement_changed / n_problems if n_problems > 0 else 0
        consensus_rate = n_consensus / n_problems if n_problems > 0 else 0
        judge_accuracy = n_judge_correct / len(problems_without_consensus) if problems_without_consensus else 0

        # Average processing time
        avg_processing_time = np.mean([m.processing_time for m in self.problems_metrics])

        # Category analysis (simplified)
        categories = defaultdict(list)
        for m in self.problems_metrics:
            cat = m.category
            categories[cat].append(m)

        category_accuracy = {
            cat: sum(1 for m in metrics if m.system_correct) / len(metrics)
            for cat, metrics in categories.items()
        }

        category_consensus = {
            cat: sum(1 for m in metrics if m.solver_consensus) / len(metrics)
            for cat, metrics in categories.items()
        }

        # Confidence analysis
        confidences_correct = []
        confidences_wrong = []

        for result in self.results:
            confidence = result.get('confidence', 0)
            system_correct = result.get('final_answer', '').strip().lower() == result.get('ground_truth',
                                                                                          '').strip().lower()

            if system_correct:
                confidences_correct.append(confidence)
            else:
                confidences_wrong.append(confidence)

        avg_confidence_when_correct = np.mean(confidences_correct) if confidences_correct else 0
        avg_confidence_when_wrong = np.mean(confidences_wrong) if confidences_wrong else 0

        # Confidence correlation
        all_confidences = []
        all_correct = []
        for result in self.results:
            confidence = result.get('confidence', 0)
            system_correct = result.get('final_answer', '').strip().lower() == result.get('ground_truth',
                                                                                          '').strip().lower()
            all_confidences.append(confidence)
            all_correct.append(1 if system_correct else 0)

        if len(all_confidences) > 1:
            confidence_correlation = np.corrcoef(all_confidences, all_correct)[0, 1]
        else:
            confidence_correlation = 0

        # Solver performance analysis
        solver_accuracy = defaultdict(list)
        solver_wins = defaultdict(int)

        for result in self.results:
            winner = result.get('final_judgement', {}).get('winner', '')
            if winner:
                solver_wins[winner] += 1

            for solver_id, solution in result.get('initial_solutions', {}).items():
                if solution.get('parse_success', True):
                    answer = str(solution.get('final_answer', '')).strip().lower()
                    correct = answer == result.get('ground_truth', '').strip().lower()
                    solver_accuracy[solver_id].append(correct)

        solver_individual_accuracy = {
            solver_id: np.mean(accuracies) if accuracies else 0
            for solver_id, accuracies in solver_accuracy.items()
        }

        # Error analysis
        error_types = defaultdict(int)
        contradiction_problems = []

        for result in self.results:
            # Count parse errors
            for solver_id, solution in result.get('initial_solutions', {}).items():
                if not solution.get('parse_success', True):
                    error_types['parse_error'] += 1

            # Check for contradictions
            answers = []
            for solver_id, solution in result.get('initial_solutions', {}).items():
                if solution.get('parse_success', True):
                    answer = solution.get('final_answer', '')
                    if answer:
                        answers.append(str(answer))

            if len(set(answers)) > 1:  # Multiple different answers
                contradiction_problems.append(result.get('problem_id', 'unknown'))

        common_error_types = sorted(error_types.items(), key=lambda x: x[1], reverse=True)

        self.system_metrics = SystemMetrics(
            overall_accuracy=overall_accuracy,
            improvement_rate=improvement_rate,
            consensus_rate=consensus_rate,
            judge_accuracy=judge_accuracy,
            avg_processing_time=avg_processing_time,
            category_accuracy=category_accuracy,
            category_consensus=category_consensus,
            avg_confidence_when_correct=avg_confidence_when_correct,
            avg_confidence_when_wrong=avg_confidence_when_wrong,
            confidence_correlation=confidence_correlation,
            solver_individual_accuracy=solver_individual_accuracy,
            solver_contribution_to_wins=solver_wins,
            common_error_types=common_error_types,
            problems_with_contradictions=contradiction_problems
        )

        return self.system_metrics

    def compare_with_baselines(self, baseline_results: Dict[str, List[bool]]) -> Dict[str, Any]:
        """
        Compare system performance with baselines

        Args:
            baseline_results: Dict with baseline names as keys and list of correctness as values

        Returns:
            Comparison metrics
        """
        system_correctness = [m.system_correct for m in self.problems_metrics]

        comparison = {
            'system_accuracy': np.mean(system_correctness) if system_correctness else 0,
            'baselines': {}
        }

        for baseline_name, correctness_list in baseline_results.items():
            if len(correctness_list) == len(system_correctness):
                baseline_accuracy = np.mean(correctness_list)
                comparison['baselines'][baseline_name] = {
                    'accuracy': baseline_accuracy,
                    'improvement_over_baseline': comparison['system_accuracy'] - baseline_accuracy,
                    'relative_improvement': (comparison[
                                                 'system_accuracy'] - baseline_accuracy) / baseline_accuracy if baseline_accuracy > 0 else 0
                }

        return comparison

    def generate_detailed_report(self, output_path: str = "data/evaluation_report.json"):
        """Generate detailed evaluation report"""
        if not self.system_metrics:
            self.calculate_system_metrics()

        report = {
            'summary': {
                'total_problems': len(self.problems_metrics),
                'overall_accuracy': self.system_metrics.overall_accuracy,
                'improvement_rate': self.system_metrics.improvement_rate,
                'consensus_rate': self.system_metrics.consensus_rate,
                'judge_accuracy': self.system_metrics.judge_accuracy,
                'avg_processing_time_seconds': self.system_metrics.avg_processing_time
            },
            'category_performance': self.system_metrics.category_accuracy,
            'solver_performance': self.system_metrics.solver_individual_accuracy,
            'confidence_analysis': {
                'avg_when_correct': self.system_metrics.avg_confidence_when_correct,
                'avg_when_wrong': self.system_metrics.avg_confidence_when_wrong,
                'correlation_with_correctness': self.system_metrics.confidence_correlation
            },
            'problem_details': [
                {
                    'problem_id': m.problem_id,
                    'category': m.category,
                    'system_correct': m.system_correct,
                    'system_answer': m.system_answer,
                    'ground_truth': m.ground_truth,
                    'solver_consensus': m.solver_consensus,
                    'refinement_changed': m.refinement_changed,
                    'judge_picked_correct': m.judge_picked_correct
                }
                for m in self.problems_metrics
            ],
            'error_analysis': {
                'common_error_types': self.system_metrics.common_error_types,
                'problems_with_contradictions': self.system_metrics.problems_with_contradictions
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Generated detailed report at {output_path}")
        return report


# Baseline experiment runner
class BaselineExperiment:
    """Runs baseline experiments for comparison"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        from src.model_clients import ModelFactory
        self.model_factory = ModelFactory()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    async def run_single_model_baseline(
            self,
            problems: List[Dict[str, Any]],
            model_key: str = "llama_3_3_70b"
    ) -> List[bool]:
        """Run single model baseline (just ask once)"""
        from src.model_clients import BatchProcessor
        from src.agent_prompts import PromptTemplates
        from src.parsing_utils import extract_final_answer

        model = self.model_factory.create_client(model_key, self.config)
        batch_processor = BatchProcessor(max_concurrent=3)

        correctness = []

        for problem in problems:
            prompt = PromptTemplates.get_baseline_prompt(problem['problem'])

            try:
                response = await model.generate_async(prompt)
                answer = extract_final_answer(response.content)
                ground_truth = problem['ground_truth_answer'].strip().lower()

                is_correct = answer.strip().lower() == ground_truth
                correctness.append(is_correct)

            except Exception as e:
                print(f"Error in baseline for {problem['id']}: {e}")
                correctness.append(False)

        return correctness

    async def run_voting_baseline(
            self,
            problems: List[Dict[str, Any]],
            model_key: str = "llama_3_3_70b",
            n_votes: int = 3
    ) -> List[bool]:
        """Run voting baseline (multiple independent answers, majority vote)"""
        from src.model_clients import BatchProcessor
        from src.agent_prompts import PromptTemplates
        from src.parsing_utils import extract_final_answer

        model = self.model_factory.create_client(model_key, self.config)
        batch_processor = BatchProcessor(max_concurrent=3)

        correctness = []

        for problem in problems:
            answers = []

            # Get multiple independent answers
            for _ in range(n_votes):
                try:
                    prompt = PromptTemplates.get_baseline_prompt(problem['problem'])
                    response = await model.generate_async(prompt)
                    answer = extract_final_answer(response.content)
                    if answer:
                        answers.append(answer.strip().lower())
                except Exception as e:
                    print(f"Error in voting baseline for {problem['id']}: {e}")

            # Majority vote
            if answers:
                from collections import Counter
                most_common = Counter(answers).most_common(1)
                if most_common:
                    final_answer = most_common[0][0]
                    ground_truth = problem['ground_truth_answer'].strip().lower()
                    is_correct = final_answer == ground_truth
                    correctness.append(is_correct)
                else:
                    correctness.append(False)
            else:
                correctness.append(False)

        return correctness


if __name__ == "__main__":
    # Example usage
    calculator = MetricsCalculator("data/results")
    calculator.load_results()

    problem_metrics = calculator.calculate_problem_metrics()
    system_metrics = calculator.calculate_system_metrics()

    print("\n=== System Performance ===")
    print(f"Overall Accuracy: {system_metrics.overall_accuracy:.2%}")
    print(f"Improvement Rate: {system_metrics.improvement_rate:.2%}")
    print(f"Consensus Rate: {system_metrics.consensus_rate:.2%}")
    print(f"Judge Accuracy: {system_metrics.judge_accuracy:.2%}")
    print(f"Avg Processing Time: {system_metrics.avg_processing_time:.2f}s")

    print("\n=== Solver Performance ===")
    for solver, accuracy in system_metrics.solver_individual_accuracy.items():
        print(f"{solver}: {accuracy:.2%}")