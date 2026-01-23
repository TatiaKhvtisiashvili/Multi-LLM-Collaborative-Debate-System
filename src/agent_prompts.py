"""
Agent Prompts - System prompts and templates for all roles and stages
Person B Responsibility
"""
from typing import Dict, Any, List
import json


class PromptTemplates:
    """All prompt templates for the debate system"""

    # Stage 0: Role Assignment Prompts
    @staticmethod
    def get_self_assessment_prompt(problem: str, available_roles: List[str]) -> str:
        return f"""You are participating in a collaborative problem-solving system. 
Analyze the following problem and assess which role you would be best suited for.

PROBLEM:
{problem}

AVAILABLE ROLES:
- Solver: Generate initial solutions with step-by-step reasoning
- Judge: Evaluate solutions and pick the best final answer

SELF-ASSESSMENT INSTRUCTIONS:
1. Analyze the problem type and required skills
2. Assess your strengths/weaknesses for this specific problem
3. Rate your confidence for each role (0.0 to 1.0)
4. Provide clear reasoning for your preference

OUTPUT FORMAT (JSON):
{{
    "role_preferences": ["Solver", "Judge"],  # Ordered by preference
    "confidence_by_role": {{
        "Solver": 0.85,
        "Judge": 0.75
    }},
    "reasoning": "I should be Solver because I'm strong at mathematical reasoning...",
    "strengths_for_this_problem": ["step-by-step reasoning", "logical deduction"],
    "weaknesses_for_this_problem": ["potential calculation errors"]
}}

Respond ONLY with valid JSON, no additional text."""

    # Stage 1: Solution Generation Prompts
    @staticmethod
    def get_solution_prompt(problem: str, solver_id: str) -> str:
        return f"""You are Solver {solver_id} in a collaborative debate system.

TASK: Generate a complete, step-by-step solution to the following problem.

IMPORTANT INSTRUCTIONS:
1. Think step by step
2. Show all reasoning clearly
3. State any assumptions you make
4. Provide a final answer
5. Be as precise and accurate as possible

PROBLEM:
{problem}

OUTPUT FORMAT (JSON):
{{
    "solver_id": "{solver_id}",
    "solution_steps": [
        {{"step": 1, "description": "First step...", "reasoning": "Why this step..."}},
        {{"step": 2, "description": "Second step...", "reasoning": "Why this step..."}}
    ],
    "final_answer": "Your final answer here",
    "confidence": 0.95,
    "assumptions": ["List assumptions if any"],
    "alternative_approaches_considered": ["Briefly mention other approaches considered"]
}}

Respond ONLY with valid JSON, no additional text."""

    # Stage 2: Peer Review Prompts
    @staticmethod
    def get_peer_review_prompt(problem: str, solution_to_review: Dict[str, Any], reviewer_id: str) -> str:
        solution_json = json.dumps(solution_to_review, indent=2)

        return f"""You are a critical peer reviewer in a collaborative debate system.

TASK: Review another solver's solution to identify strengths, weaknesses, and errors.

PROBLEM:
{problem}

SOLUTION TO REVIEW:
{solution_json}

REVIEW GUIDELINES:
1. Check for logical errors or leaps in reasoning
2. Identify calculation mistakes
3. Note missing steps or assumptions
4. Consider edge cases not addressed
5. Assess overall correctness and completeness

Be constructive but rigorous. If something is wrong, explain why clearly.

OUTPUT FORMAT (JSON):
{{
    "reviewer_id": "{reviewer_id}",
    "solution_being_reviewed": "{solution_to_review.get('solver_id', 'unknown')}",
    "evaluation": {{
        "strengths": ["Clear step 1-3", "Correct formula application", ...],
        "weaknesses": ["Step 5 makes unjustified leap", "Missing edge case X", ...],
        "errors": [
            {{
                "location": "Step 5",
                "error_type": "logical_error",  # or "calculation_error", "assumption_error", "missing_step"
                "description": "Claims X implies Y but this is false when Z...",
                "severity": "critical"  # or "minor", "moderate"
            }}
        ],
        "suggested_changes": [
            "Reconsider step 5 with counterexample...",
            "Add verification for case when n=0"
        ]
    }},
    "overall_assessment": "correct",  # or "mostly_correct", "partially_correct", "flawed", "incorrect"
    "confidence_in_assessment": 0.90
}}

Respond ONLY with valid JSON, no additional text."""

    # Stage 3: Refinement Prompts
    @staticmethod
    def get_refinement_prompt(
            problem: str,
            original_solution: Dict[str, Any],
            reviews: List[Dict[str, Any]],
            solver_id: str
    ) -> str:
        reviews_json = json.dumps(reviews, indent=2)

        return f"""You are Solver {solver_id} in a collaborative debate system.

TASK: Refine your original solution based on peer feedback.

PROBLEM:
{problem}

YOUR ORIGINAL SOLUTION:
{json.dumps(original_solution, indent=2)}

PEER REVIEWS (2 reviews):
{reviews_json}

INSTRUCTIONS:
1. Carefully consider each critique from the reviews
2. Address valid points by revising your solution
3. Defend your reasoning if critiques are incorrect
4. Produce a refined final solution
5. Explicitly state what changes you made and why

OUTPUT FORMAT (JSON):
{{
    "solver_id": "{solver_id}",
    "changes_made": [
        {{
            "critique_summary": "Step 5 was wrong",
            "response": "Fixed by correcting the formula...",
            "accepted": true,
            "explanation": "The reviewer correctly identified an error in..."
        }},
        {{
            "critique_summary": "Missing edge case",
            "response": "This case doesn't apply because...",
            "accepted": false,
            "explanation": "The edge case mentioned is already covered by assumption X..."
        }}
    ],
    "refined_solution": {{
        "solution_steps": [...],  # Full refined solution steps
        "improvements_from_original": ["Fixed step 5", "Added verification for edge case"]
    }},
    "refined_answer": "Updated final answer if changed",
    "confidence": 0.95,
    "summary_of_changes": "Brief summary of key changes made"
}}

Respond ONLY with valid JSON, no additional text."""

    # Stage 4: Judgement Prompts
    @staticmethod
    def get_judgement_prompt(
            problem: str,
            original_solutions: List[Dict[str, Any]],
            all_reviews: List[Dict[str, Any]],
            refined_solutions: List[Dict[str, Any]]
    ) -> str:
        data = {
            "problem": problem,
            "original_solutions": original_solutions,
            "all_reviews": all_reviews,
            "refined_solutions": refined_solutions
        }
        data_json = json.dumps(data, indent=2)

        return f"""You are the Final Judge in a collaborative debate system.

    TASK: Evaluate all solutions and reviews to select the best final answer.

    PROBLEM:
    {problem}

    COMPLETE DEBATE DATA:
    {data_json}

    JUDGING CRITERIA:
    1. Correctness and accuracy of final answer
    2. Quality and clarity of reasoning
    3. Responsiveness to peer feedback
    4. Robustness (handles edge cases)
    5. Efficiency and elegance of solution

    IMPORTANT: 
    - You must select ONE winner from the debaters' solutions
    - Consider both original and refined solutions
    - Do NOT provide your own answer - pick one from the debaters
    - Base your decision on the quality of reasoning, not external knowledge

    OUTPUT FORMAT (JSON):
    {{
        "judgement": {{
            "winner": "solver_1",  # or "solver_2", "solver_3"
            "winner_original_id": "solver_1",
            "confidence": 0.85,
            "reasoning": "Solver 1's solution is strongest because...",
            "ranking": [
                {{"solver": "solver_1", "score": 0.95, "reason": "Most accurate with best reasoning"}},
                {{"solver": "solver_3", "score": 0.80, "reason": "Good but missed edge case"}},
                {{"solver": "solver_2", "score": 0.65, "reason": "Fundamental error in approach"}}
            ],
            "key_differentiators": [
                "Solver 1 correctly handled the edge case that others missed",
                "Solver 3's refinement showed good learning but initial solution was weak"
            ]
        }},
        "selected_final_answer": "The final answer to return to user (MUST be one of the debaters' answers)"
    }}

    Respond ONLY with valid JSON, no additional text."""

    # Baseline prompts for comparison
    @staticmethod
    def get_baseline_prompt(problem: str) -> str:
        return f"""Solve the following problem. Show your reasoning step by step and provide a final answer.

PROBLEM:
{problem}

Respond with a clear, well-reasoned solution ending with "Final answer: [your answer]"."""


class SystemPrompts:
    """System prompts for different roles"""

    SOLVER_SYSTEM_PROMPT = """You are an expert problem solver participating in a collaborative debate. 
Your goal is to solve problems accurately, learn from peers, and improve through feedback.
Be precise, logical, and thorough in your reasoning."""

    REVIEWER_SYSTEM_PROMPT = """You are a rigorous peer reviewer. Your goal is to identify errors, 
oversights, and areas for improvement in others' solutions. Be constructive but uncompromising 
on accuracy and logical rigor."""

    JUDGE_SYSTEM_PROMPT = """You are the final arbitrator in a collaborative debate system. 
Your goal is to objectively evaluate all solutions and select the best one based on 
correctness, reasoning quality, and responsiveness to feedback. Be impartial and thorough."""

    SELF_ASSESSMENT_SYSTEM_PROMPT = """You are self-aware AI assessing your own capabilities 
for a specific problem. Be honest and accurate about your strengths and weaknesses."""


class OutputValidator:
    """Validates and cleans LLM outputs"""

    @staticmethod
    def extract_json_from_text(text: str) -> Dict[str, Any]:
        """Extract JSON from text that might have extra content"""
        # Find JSON-like content
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in response")

        json_str = text[start_idx:end_idx]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            json_str = OutputValidator._fix_json(json_str)
            return json.loads(json_str)

    @staticmethod
    def _fix_json(json_str: str) -> str:
        """Attempt to fix common JSON issues"""
        # Remove trailing commas
        import re
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Fix missing quotes on keys
        def fix_missing_quotes(match):
            key = match.group(1)
            if not key.startswith('"'):
                return f'"{key}":'
            return match.group(0)

        json_str = re.sub(r'(\w+)\s*:', fix_missing_quotes, json_str)

        return json_str

    @staticmethod
    def validate_solution_structure(data: Dict[str, Any]) -> bool:
        """Validate solution has required fields"""
        required = ['solver_id', 'solution_steps', 'final_answer', 'confidence']
        return all(field in data for field in required)

    @staticmethod
    def validate_review_structure(data: Dict[str, Any]) -> bool:
        """Validate review has required fields"""
        required = ['reviewer_id', 'evaluation', 'overall_assessment']
        if not all(field in data for field in required):
            return False

        # Check evaluation subfields
        eval_fields = ['strengths', 'weaknesses', 'errors', 'suggested_changes']
        return all(field in data['evaluation'] for field in eval_fields)