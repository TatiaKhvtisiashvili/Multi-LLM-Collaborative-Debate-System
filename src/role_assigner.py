"""
Role Assigner - Implements Stage 0 & 0.5: Self-assessment and algorithmic assignment
Person B Responsibility
"""
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import re


@dataclass
class SelfAssessment:
    """Structured self-assessment from an LLM"""
    role_preferences: List[str]
    confidence_by_role: Dict[str, float]
    reasoning: str
    strengths_for_this_problem: List[str]
    weaknesses_for_this_problem: List[str]
    raw_response: Dict[str, Any]


class RoleAssigner:
    """Handles role assignment based on self-assessments"""

    def __init__(self):
        # Model capabilities mapping (can be extended)
        self.capability_keywords = {
            "mathematical": ["math", "calculate", "formula", "probability", "statistics"],
            "logical": ["logic", "reasoning", "deduction", "inference"],
            "analytical": ["analyze", "evaluate", "assess", "compare"],
            "creative": ["creative", "novel", "innovative", "alternative"],
            "critical": ["critical", "review", "evaluate", "critique"],
            "judicial": ["judge", "arbitrate", "decide", "evaluate"]
        }

    def parse_self_assessment(self, response_text: str, model_name: str) -> SelfAssessment:
        """Parse LLM self-assessment response"""
        try:
            # Extract JSON from response
            if isinstance(response_text, dict):
                data = response_text
            else:
                # Find JSON in text
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")

            # Ensure required fields
            required = ['role_preferences', 'confidence_by_role', 'reasoning']
            for field in required:
                if field not in data:
                    data[field] = None

            # Ensure confidence_by_role has all roles
            if 'confidence_by_role' not in data or not data['confidence_by_role']:
                data['confidence_by_role'] = {
                    'Solver': 0.5,
                    'Judge': 0.5
                }

            return SelfAssessment(
                role_preferences=data.get('role_preferences', ['Solver', 'Judge']),
                confidence_by_role=data['confidence_by_role'],
                reasoning=data.get('reasoning', ''),
                strengths_for_this_problem=data.get('strengths_for_this_problem', []),
                weaknesses_for_this_problem=data.get('weaknesses_for_this_problem', []),
                raw_response=data
            )
        except Exception as e:
            print(f"Error parsing self-assessment: {e}")
            # Return default assessment
            return SelfAssessment(
                role_preferences=['Solver', 'Judge'],
                confidence_by_role={'Solver': 0.5, 'Judge': 0.5},
                reasoning='Failed to parse assessment',
                strengths_for_this_problem=[],
                weaknesses_for_this_problem=[],
                raw_response={}
            )

    def analyze_problem_type(self, problem_text: str) -> Dict[str, float]:
        """Analyze problem to determine needed capabilities"""
        problem_lower = problem_text.lower()

        scores = {}
        for capability, keywords in self.capability_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in problem_lower:
                    score += 1
            scores[capability] = min(score / len(keywords), 1.0)

        return scores

    def algorithmic_assignment(
            self,
            assessments: Dict[str, SelfAssessment],
            problem_text: str,
            model_names: List[str]
    ) -> Dict[str, str]:
        """
        Deterministic role assignment algorithm
        Returns mapping: {solver_1: model_name, solver_2: ..., solver_3: ..., judge: ...}
        """
        # Step 1: Problem analysis
        problem_needs = self.analyze_problem_type(problem_text)

        # Step 2: Score each model for each role
        model_scores = {}

        for model_name in model_names:
            assessment = assessments.get(model_name)
            if not assessment:
                # Default scores if no assessment
                model_scores[model_name] = {
                    'Solver': 0.5,
                    'Judge': 0.5,
                    'raw_confidence': {'Solver': 0.5, 'Judge': 0.5}
                }
                continue

            # Base score from self-confidence
            solver_score = assessment.confidence_by_role.get('Solver', 0.5)
            judge_score = assessment.confidence_by_role.get('Judge', 0.5)

            # Adjust based on problem needs and self-reported strengths
            reasoning_lower = assessment.reasoning.lower()

            # Boost solver score for mathematical/logical problems
            if problem_needs.get('mathematical', 0) > 0.5:
                if any(word in reasoning_lower for word in ['math', 'calculate', 'formula']):
                    solver_score *= 1.2

            # Boost judge score for analytical/critical problems
            if problem_needs.get('analytical', 0) > 0.5 or problem_needs.get('critical', 0) > 0.5:
                if any(word in reasoning_lower for word in ['analyze', 'evaluate', 'critical']):
                    judge_score *= 1.2

            # Cap scores at 1.0
            solver_score = min(solver_score, 1.0)
            judge_score = min(judge_score, 1.0)

            model_scores[model_name] = {
                'Solver': solver_score,
                'Judge': judge_score,
                'raw_confidence': assessment.confidence_by_role
            }

        # Step 3: Assign roles using Hungarian algorithm (simplified)
        # For 4 models and 4 roles (3 solvers, 1 judge)
        assignment = self._assign_roles_hungarian(model_scores)

        return assignment

    def _assign_roles_hungarian(self, model_scores: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Simplified Hungarian assignment for small set"""
        models = list(model_scores.keys())

        # We need 3 solvers and 1 judge
        # First, pick judge - highest judge score
        judge_scores = [(model, model_scores[model]['Judge']) for model in models]
        judge_scores.sort(key=lambda x: x[1], reverse=True)
        judge_model = judge_scores[0][0]

        # Remaining models become solvers
        remaining_models = [m for m in models if m != judge_model]

        # Sort solvers by solver score (highest first)
        solver_scores = [(model, model_scores[model]['Solver']) for model in remaining_models]
        solver_scores.sort(key=lambda x: x[1], reverse=True)

        # Assign solver positions (solver_1 gets highest score)
        assignment = {
            'judge': judge_model,
            'solver_1': solver_scores[0][0] if len(solver_scores) > 0 else None,
            'solver_2': solver_scores[1][0] if len(solver_scores) > 1 else None,
            'solver_3': solver_scores[2][0] if len(solver_scores) > 2 else None
        }

        # If we have fewer than 4 models, some roles might be None
        # In that case, assign duplicate models (real system would have 4+ models)
        assigned_models = set(v for v in assignment.values() if v)
        all_models = set(models)

        # Fill any missing assignments with best available model
        if None in assignment.values():
            available_models = list(all_models - assigned_models)
            for role, model in assignment.items():
                if model is None and available_models:
                    assignment[role] = available_models.pop(0)

        return assignment

    def get_role_descriptions(self) -> Dict[str, str]:
        """Get detailed role descriptions for prompts"""
        return {
            "Solver": """As a Solver, your primary responsibility is to:
1. Generate accurate, step-by-step solutions to the given problem
2. Provide clear reasoning for each step
3. State all assumptions explicitly
4. Offer a final answer with confidence assessment
5. Be prepared to revise based on peer feedback""",

            "Judge": """As the Final Judge, your primary responsibility is to:
1. Evaluate all solutions objectively
2. Consider both original and refined versions
3. Assess quality of reasoning and responsiveness to feedback
4. Select the single best solution
5. Provide detailed justification for your decision"""
        }


# Utility for tracking assignments
class AssignmentTracker:
    """Tracks and logs role assignments"""

    def __init__(self, log_file: str = "data/role_assignments.jsonl"):
        self.log_file = log_file
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_assignment(
            self,
            problem_id: str,
            assignments: Dict[str, str],
            assessments: Dict[str, SelfAssessment],
            problem_needs: Dict[str, float]
    ):
        """Log assignment decision"""
        log_entry = {
            "problem_id": problem_id,
            "timestamp": self._get_timestamp(),
            "assignments": assignments,
            "assessments": {
                model: {
                    "role_preferences": ass.role_preferences,
                    "confidence_by_role": ass.confidence_by_role,
                    "reasoning_summary": ass.reasoning[:200] + "..." if len(ass.reasoning) > 200 else ass.reasoning
                }
                for model, ass in assessments.items()
            },
            "problem_needs": problem_needs,
            "assignment_rationale": self._generate_rationale(assignments, assessments, problem_needs)
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()

    def _generate_rationale(
            self,
            assignments: Dict[str, str],
            assessments: Dict[str, SelfAssessment],
            problem_needs: Dict[str, float]
    ) -> str:
        """Generate human-readable rationale for assignment"""
        rationale = []

        # Judge rationale
        judge_model = assignments.get('judge')
        if judge_model and judge_model in assessments:
            judge_conf = assessments[judge_model].confidence_by_role.get('Judge', 0)
            rationale.append(f"Assigned {judge_model} as Judge (confidence: {judge_conf:.2f})")

        # Solver rationales
        for i in range(1, 4):
            role = f'solver_{i}'
            solver_model = assignments.get(role)
            if solver_model and solver_model in assessments:
                solver_conf = assessments[solver_model].confidence_by_role.get('Solver', 0)
                rationale.append(f"Assigned {solver_model} as Solver_{i} (confidence: {solver_conf:.2f})")

        # Problem needs summary
        if problem_needs:
            primary_needs = [k for k, v in problem_needs.items() if v > 0.5]
            if primary_needs:
                rationale.append(f"Problem requires: {', '.join(primary_needs)}")

        return "; ".join(rationale)


# Example usage
if __name__ == "__main__":
    # Test the role assigner
    assigner = RoleAssigner()

    # Mock assessments
    mock_assessments = {
        "llama_3_3_70b": SelfAssessment(
            role_preferences=["Solver", "Judge"],
            confidence_by_role={"Solver": 0.85, "Judge": 0.70},
            reasoning="I'm strong at mathematical reasoning and step-by-step problem solving",
            strengths_for_this_problem=["mathematical reasoning", "logical deduction"],
            weaknesses_for_this_problem=["potential overconfidence in calculations"],
            raw_response={}
        ),
        "gemini_2_flash": SelfAssessment(
            role_preferences=["Judge", "Solver"],
            confidence_by_role={"Solver": 0.75, "Judge": 0.80},
            reasoning="I'm good at evaluating multiple solutions and identifying key differences",
            strengths_for_this_problem=["analytical comparison", "error detection"],
            weaknesses_for_this_problem=["may miss subtle mathematical nuances"],
            raw_response={}
        )
    }

    problem = "In how many ways can you tile a 3×8 rectangle with 2×1 dominoes?"

    assignments = assigner.algorithmic_assignment(
        mock_assessments,
        problem,
        ["llama_3_3_70b", "gemini_2_flash", "mistral_latest", "gemini_2_5_pro"]
    )

    print("Role Assignments:")
    for role, model in assignments.items():
        print(f"  {role}: {model}")