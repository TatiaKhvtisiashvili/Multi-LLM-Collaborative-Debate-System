"""
Dataset Manager - Handles problem dataset creation and management
Person A Responsibility
"""
import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import hashlib
import logging
import os


@dataclass
class Problem:
    """Problem data structure"""
    id: str
    category: str
    problem: str
    ground_truth_answer: str
    ground_truth_reasoning: str
    difficulty: str = "medium"
    source: str = "custom"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class DatasetManager:
    def __init__(self, filepath: str = "data/problems.json"):
        self.filepath = filepath
        self.problems = []
        self.logger = logging.getLogger(__name__)  # Add logger

    def generate_problem_id(self, category: str, index: int) -> str:
        """Generate consistent problem ID"""
        prefix = {
            "Mathematical/Logical Reasoning": "MATH",
            "Physics & Scientific Reasoning": "PHYS",
            "Logic Puzzles & Constraint Satisfaction": "LOGIC",
            "Strategic Game Theory": "GAME"
        }.get(category, "PROB")
        return f"{prefix}_{index:02d}"

    def load_dataset(self):
        """Load the problem dataset from the configured path."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()

                if not content:
                    self.logger.warning(f"Dataset file {self.filepath} is empty")
                    return []

                if content.startswith('{'):
                    # Single JSON object (dictionary of problems)
                    data = json.loads(content)
                    problems = []

                    for problem_id, problem_data in data.items():
                        # Ensure all required fields exist
                        problem_data_with_id = problem_data.copy()
                        problem_data_with_id['id'] = problem_id

                        # Add ground_truth_reasoning if missing
                        if 'ground_truth_reasoning' not in problem_data_with_id:
                            problem_data_with_id['ground_truth_reasoning'] = (
                                f"See answer: {problem_data_with_id.get('ground_truth_answer', '')}"
                            )

                        # Create Problem object
                        problem = Problem(
                            id=problem_data_with_id['id'],
                            category=problem_data_with_id.get('category', 'Mathematical/Logical Reasoning'),
                            problem=problem_data_with_id['problem'],
                            ground_truth_answer=problem_data_with_id['ground_truth_answer'],
                            ground_truth_reasoning=problem_data_with_id['ground_truth_reasoning'],
                            difficulty=problem_data_with_id.get('difficulty', 'medium'),
                            source=problem_data_with_id.get('source', 'custom')
                        )
                        problems.append(problem)

                    self.problems = problems
                    return problems
                else:
                    # JSONL format (one JSON per line)
                    problems = []
                    for line in content.split('\n'):
                        if line.strip():
                            problem_data = json.loads(line.strip())
                            # Ensure ID exists
                            if 'id' not in problem_data:
                                problem_data['id'] = f"UNKNOWN_{hashlib.md5(line.encode()).hexdigest()[:8]}"

                            problem = Problem(**problem_data)
                            problems.append(problem)

                    self.problems = problems
                    return problems

        except FileNotFoundError:
            self.logger.warning(f"Dataset file {self.filepath} not found")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in dataset file {self.filepath}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading dataset from {self.filepath}: {e}")
            raise

    def create_curated_problems(self) -> List[Problem]:
        """Create 15 curated problems (Person A task)"""
        curated_problems = [
            # Your existing curated problems...
            Problem(
                id="MATH_01",
                category="Mathematical/Logical Reasoning",
                problem="In how many ways can you tile a 3×8 rectangle with 2×1 dominoes?",
                ground_truth_answer="34",
                ground_truth_reasoning="This is a Fibonacci sequence problem where F_n = F_{n-1} + F_{n-2} with F_1=1, F_2=1. For width n=8, F_8 = 34."
            ),
            # ... add other problems
        ]
        return curated_problems[:15]

    def load_or_create_default_dataset(self) -> List[Problem]:
        """Load dataset or create default if doesn't exist"""
        try:
            problems = self.load_dataset()

            if not problems:
                self.logger.info("Dataset not found. Creating default dataset...")
                problems = self.create_curated_problems()

                if len(problems) < 25:
                    # Add some additional curated problems to reach 25
                    additional = [
                        Problem(
                            id="MATH_08",
                            category="Mathematical/Logical Reasoning",
                            problem="How many trailing zeros in 100!?",
                            ground_truth_answer="24",
                            ground_truth_reasoning="Count factors of 5: ⌊100/5⌋+⌊100/25⌋ = 20+4 = 24.",
                            difficulty="medium"
                        ),
                        # Add more as needed...
                    ]
                    problems.extend(additional[:25 - len(problems)])

                self.save_dataset(problems)
                self.logger.info(f"Created default dataset with {len(problems)} problems")

            return problems

        except Exception as e:
            self.logger.error(f"Error in load_or_create_default_dataset: {e}")
            # Return empty list on error
            return []

    def save_dataset(self, problems: List[Problem]):
        """Save problems to file"""
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

            # Convert to dictionary format for saving
            data = {}
            for problem in problems:
                data[problem.id] = {
                    'category': problem.category,
                    'problem': problem.problem,
                    'ground_truth_answer': problem.ground_truth_answer,
                    'ground_truth_reasoning': problem.ground_truth_reasoning,
                    'difficulty': problem.difficulty,
                    'source': problem.source
                }

            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved {len(problems)} problems to {self.filepath}")

        except Exception as e:
            self.logger.error(f"Error saving dataset to {self.filepath}: {e}")
            raise

    def validate_dataset(self, problems: List[Problem]) -> bool:
        """Validate dataset has all required fields"""
        if not problems:
            self.logger.warning("Dataset is empty")
            return False

        if len(problems) != 25:
            self.logger.warning(f"Expected 25 problems, got {len(problems)}")

        for p in problems:
            required_fields = ['id', 'category', 'problem', 'ground_truth_answer', 'ground_truth_reasoning']
            for field in required_fields:
                if not getattr(p, field):
                    self.logger.error(f"Problem {p.id} missing {field}")
                    return False

        return True


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    manager = DatasetManager("data/problems.json")

    # Load or create dataset
    all_problems = manager.load_or_create_default_dataset()

    if manager.validate_dataset(all_problems):
        print(f"✓ Dataset ready with {len(all_problems)} problems")
        for p in all_problems[:3]:  # Show first 3
            print(f"  - {p.id}: {p.problem[:50]}...")
    else:
        print("Dataset validation failed")