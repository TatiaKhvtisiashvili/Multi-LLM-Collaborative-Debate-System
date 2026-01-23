"""
Dataset Generation Script - Creates the 25-problem dataset
Person B can use this to generate problems with LLMs
"""
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import asyncio

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset_manager import DatasetManager, Problem
from src.model_clients import ModelFactory, ResponseCache
from src.parsing_utils import LLMOutputParser, extract_final_answer
from src.utils import setup_logging, Timer


class DatasetGenerator:
    """Generates problems using LLMs"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.logger = setup_logging()
        self.parser = LLMOutputParser()

        # Initialize a model client for generation
        # Using a free model via OpenRouter
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_factory = ModelFactory()
        self.generation_model = model_factory.create_client(
            "llama_3_3_70b", config
        )

        self.cache = ResponseCache()

    async def generate_problem(
            self,
            category: str,
            difficulty: str = "medium"
    ) -> Dict[str, Any]:
        """Generate a single problem using LLM"""

        prompt = self._get_generation_prompt(category, difficulty)
        system_prompt = """You are an expert problem creator. Generate original, 
        challenging problems with verifiable answers. Be creative but ensure the 
        problem is well-defined and has a single correct answer."""

        try:
            # Check cache first
            cached = self.cache.get(
                self.generation_model.model_name, prompt, system_prompt
            )

            if cached:
                self.logger.debug("Using cached generation")
                response = cached.content
            else:
                response_obj = await self.generation_model.generate_async(
                    prompt, system_prompt
                )
                response = response_obj.content
                self.cache.set(
                    self.generation_model.model_name,
                    prompt,
                    system_prompt,
                    response_obj
                )

            # Parse the response
            # Expected format: JSON with problem, answer, reasoning
            try:
                # Try to extract JSON
                parsed = self.parser.parse_response(response, "json")

                # Validate required fields
                required = ['problem', 'answer', 'reasoning']
                if all(field in parsed for field in required):
                    return {
                        'category': category,
                        'difficulty': difficulty,
                        'problem': parsed['problem'],
                        'ground_truth_answer': str(parsed['answer']),
                        'ground_truth_reasoning': parsed['reasoning'],
                        'source': 'llm_generated',
                        'metadata': {
                            'generation_model': self.generation_model.model_name,
                            'difficulty': difficulty
                        }
                    }
                else:
                    self.logger.warning("Missing fields in generated problem")

            except:
                # Fallback: extract from text
                problem_match = self._extract_problem_from_text(response)
                if problem_match:
                    return problem_match

            # If we get here, generation failed
            return self._create_fallback_problem(category, difficulty)

        except Exception as e:
            self.logger.error(f"Error generating problem: {e}")
            return self._create_fallback_problem(category, difficulty)

    def _get_generation_prompt(self, category: str, difficulty: str) -> str:
        """Get prompt for problem generation"""

        difficulty_descriptions = {
            "easy": "straightforward, suitable for beginners",
            "medium": "challenging but solvable with careful reasoning",
            "hard": "very difficult, requiring advanced concepts"
        }

        category_examples = {
            "Mathematical/Logical Reasoning": """
            Example: "In how many ways can you tile a 3×8 rectangle with 2×1 dominoes?"
            Required: Step-by-step logical or mathematical solution.
            """,
            "Physics & Scientific Reasoning": """
            Example: "A ladder leans against a frictionless wall. Derive the minimum coefficient of friction needed with the ground to prevent slipping."
            Required: Application of physical laws and formulas.
            """,
            "Logic Puzzles & Constraint Satisfaction": """
            Example: "Five people of different nationalities live in five colored houses. Given 15 clues about their pets, drinks, and cigarette brands, who owns the fish?"
            Required: Logical deduction from constraints.
            """,
            "Strategic Game Theory": """
            Example: "In a two-player auction where bids are sealed and highest bidder pays the second-highest bid, what's the optimal bidding strategy?"
            Required: Strategic reasoning about optimal decisions.
            """
        }

        return f"""Generate an original {difficulty} difficulty problem in the category: {category}

{difficulty_descriptions.get(difficulty, 'challenging')}

{category_examples.get(category, '')}

REQUIREMENTS:
1. The problem must be self-contained and clearly stated
2. It must have exactly one correct answer
3. The answer should be verifiable (not subjective)
4. Provide the reasoning steps to reach the answer
5. Make it challenging but fair

OUTPUT FORMAT (JSON):
{{
    "problem": "The full problem statement here...",
    "answer": "The exact correct answer (e.g., '42', 'true', 'solver_1')",
    "reasoning": "Step-by-step reasoning to reach the answer...",
    "hints": ["Optional hint 1", "Optional hint 2"],
    "common_mistakes": ["Common error 1", "Common error 2"]
}}

Generate a truly original problem, not a variation of well-known puzzles."""

    def _extract_problem_from_text(self, text: str) -> Dict[str, Any]:
        """Extract problem from free text response"""
        # Simple extraction heuristics
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if len(lines) >= 3:
            # Look for problem pattern
            problem = lines[0]
            answer = None
            reasoning = "\n".join(lines[1:])

            # Try to extract answer
            for line in lines:
                lower_line = line.lower()
                if "answer:" in lower_line or "solution:" in lower_line:
                    answer = line.split(":", 1)[1].strip()
                    break

            if answer:
                return {
                    'category': 'unknown',
                    'difficulty': 'medium',
                    'problem': problem,
                    'ground_truth_answer': answer,
                    'ground_truth_reasoning': reasoning,
                    'source': 'llm_generated_extracted'
                }

        return None

    def _create_fallback_problem(self, category: str, difficulty: str) -> Dict[str, Any]:
        """Create a fallback problem if generation fails"""
        fallback_problems = {
            "Mathematical/Logical Reasoning": {
                "problem": "What is the sum of the first 100 positive integers?",
                "answer": "5050",
                "reasoning": "Use formula n(n+1)/2 with n=100: 100*101/2 = 5050."
            },
            "Physics & Scientific Reasoning": {
                "problem": "If a ball is dropped from a height of 20 meters, how long does it take to hit the ground? (g = 9.8 m/s²)",
                "answer": "2.02",
                "reasoning": "Use h = 1/2 gt². t = √(2h/g) = √(40/9.8) ≈ 2.02 seconds."
            },
            "Logic Puzzles & Constraint Satisfaction": {
                "problem": "If all A are B, and some B are C, can we conclude that some A are C?",
                "answer": "No",
                "reasoning": "Venn diagram shows A entirely within B, and C overlapping only part of B. A might not overlap with C at all."
            },
            "Strategic Game Theory": {
                "problem": "In the prisoner's dilemma, what is the dominant strategy for a rational player?",
                "answer": "Defect",
                "reasoning": "Regardless of the other player's choice, defecting yields a better or equal payoff."
            }
        }

        fallback = fallback_problems.get(category, fallback_problems["Mathematical/Logical Reasoning"])

        return {
            'category': category,
            'difficulty': difficulty,
            'problem': fallback['problem'] + f" (Fallback for {category})",
            'ground_truth_answer': fallback['answer'],
            'ground_truth_reasoning': fallback['reasoning'],
            'source': 'fallback',
            'note': 'LLM generation failed, using fallback problem'
        }

    async def generate_category_problems(
            self,
            category: str,
            count: int,
            difficulties: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate multiple problems for a category"""
        if difficulties is None:
            difficulties = ['easy', 'medium', 'hard']

        problems = []
        tasks = []

        # Distribute difficulties
        for i in range(count):
            difficulty = difficulties[i % len(difficulties)]
            tasks.append(self.generate_problem(category, difficulty))

        # Generate in parallel (but respect rate limits)
        self.logger.info(f"Generating {count} problems for {category}...")

        # Process with small batches to avoid rate limits
        batch_size = 3
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error in batch generation: {result}")
                    # Add fallback
                    problems.append(
                        self._create_fallback_problem(
                            category,
                            difficulties[(i + j) % len(difficulties)]
                        )
                    )
                elif result:
                    problems.append(result)

            # Small delay between batches
            if i + batch_size < len(tasks):
                await asyncio.sleep(2)

        return problems[:count]  # Ensure exact count

    def save_generated_problems(
            self,
            problems: List[Dict[str, Any]],
            output_file: str = "data/generated_problems.json"
    ):
        """Save generated problems to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'generated_at': self._get_timestamp(),
            'total_problems': len(problems),
            'problems': problems
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(problems)} generated problems to {output_path}")

        # Also save as individual entries for dataset manager
        self._save_for_dataset_manager(problems)

    def _save_for_dataset_manager(self, problems: List[Dict[str, Any]]):
        """Save in format compatible with DatasetManager"""
        manager = DatasetManager()

        dataset_problems = []
        for i, problem in enumerate(problems):
            p = Problem(
                id=f"GEN_{(i + 16):02d}",  # Continuing from curated problems
                category=problem['category'],
                problem=problem['problem'],
                ground_truth_answer=problem['ground_truth_answer'],
                ground_truth_reasoning=problem['ground_truth_reasoning'],
                difficulty=problem.get('difficulty', 'medium'),
                source=problem.get('source', 'llm_generated')
            )
            dataset_problems.append(p)

        # Append to existing dataset if it exists
        existing = manager.load_dataset()
        if existing:
            all_problems = existing + dataset_problems
            # Ensure we have exactly 25
            if len(all_problems) > 25:
                all_problems = all_problems[:25]
            manager.save_dataset(all_problems)
            self.logger.info(f"Updated dataset with {len(dataset_problems)} generated problems")
        else:
            self.logger.warning("No existing dataset found. Run dataset manager first.")

    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()


async def main():
    """Main function to generate dataset"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate problem dataset using LLMs')
    parser.add_argument('--count', type=int, default=10, help='Number of problems to generate')
    parser.add_argument('--categories', type=str, nargs='+',
                        default=['Mathematical/Logical Reasoning',
                                 'Physics & Scientific Reasoning',
                                 'Logic Puzzles & Constraint Satisfaction',
                                 'Strategic Game Theory'],
                        help='Categories to generate problems for')
    parser.add_argument('--output', type=str, default='data/generated_problems.json',
                        help='Output file path')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    print("Dataset Generator")
    print("=" * 50)

    generator = DatasetGenerator(config_path=args.config)

    all_problems = []

    # Distribute counts across categories
    base_count = args.count // len(args.categories)
    remainder = args.count % len(args.categories)

    with Timer() as timer:
        for i, category in enumerate(args.categories):
            count = base_count + (1 if i < remainder else 0)
            if count > 0:
                print(f"\nGenerating {count} problems for: {category}")
                category_problems = await generator.generate_category_problems(
                    category, count
                )
                all_problems.extend(category_problems)
                print(f"✓ Generated {len(category_problems)} problems")

    # Save results
    generator.save_generated_problems(all_problems, args.output)

    # Summary
    print(f"\nGeneration complete!")
    print(f"   Total problems: {len(all_problems)}")
    print(f"   Time elapsed: {timer.get_formatted()}")
    print(f"   Output file: {args.output}")

    # Category breakdown
    from collections import Counter
    categories = Counter(p['category'] for p in all_problems)
    print(f"\nCategory breakdown:")
    for category, count in categories.items():
        print(f"   {category}: {count} problems")


if __name__ == "__main__":
    asyncio.run(main())