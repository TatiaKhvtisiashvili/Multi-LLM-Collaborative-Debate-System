"""
Role Assigner - Enhanced with Collaborative Judge Selection
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


@dataclass
class JudgeVote:
    """Vote for who should be judge"""
    voter_model: str
    votes_for: Dict[str, float]  # model_name -> vote_weight
    reasoning: str


class RoleAssigner:
    """Handles role assignment with collaborative judge selection"""

    def __init__(self):
        # Model capabilities mapping
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

    def parse_judge_vote(self, response_text: str, voter_model: str) -> JudgeVote:
        """Parse vote for judge selection"""
        try:
            if isinstance(response_text, dict):
                data = response_text
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")

            return JudgeVote(
                voter_model=voter_model,
                votes_for=data.get('votes_for', {}),
                reasoning=data.get('reasoning', '')
            )
        except Exception as e:
            print(f"Error parsing judge vote from {voter_model}: {e}")
            return JudgeVote(
                voter_model=voter_model,
                votes_for={},
                reasoning='Failed to parse vote'
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

    def collaborative_judge_selection(
            self,
            assessments: Dict[str, SelfAssessment],
            model_names: List[str],
            vote_responses: Dict[str, JudgeVote] = None
    ) -> str:
        """
        Collaborative voting to select judge

        Args:
            assessments: Self-assessments from all models
            model_names: List of all model names
            vote_responses: Optional pre-parsed votes (if None, uses assessments only)

        Returns:
            Selected judge model name
        """
        # Score each model for judge role
        judge_scores = {}

        for model_name in model_names:
            score = 0.0

            # Self-assessment score (30% weight)
            if model_name in assessments:
                self_confidence = assessments[model_name].confidence_by_role.get('Judge', 0.5)
                score += self_confidence * 0.3

            # Peer votes (70% weight if available)
            if vote_responses:
                peer_votes = 0
                vote_count = 0

                for voter_model, vote in vote_responses.items():
                    if voter_model != model_name:  # Can't vote for self
                        vote_value = vote.votes_for.get(model_name, 0)
                        peer_votes += vote_value
                        vote_count += 1

                if vote_count > 0:
                    avg_peer_vote = peer_votes / vote_count
                    score += avg_peer_vote * 0.7
            else:
                # No votes - use only self-assessment
                score = self_confidence

            judge_scores[model_name] = score

        # Select highest scoring model
        selected_judge = max(judge_scores.items(), key=lambda x: x[1])[0]

        print(f"\n🗳️  Judge Selection Voting Results:")
        for model, score in sorted(judge_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"   {model}: {score:.3f}")
        print(f"   → Selected: {selected_judge}\n")

        return selected_judge

    def algorithmic_assignment(
            self,
            assessments: Dict[str, SelfAssessment],
            problem_text: str,
            model_names: List[str],
            selected_judge: str = None
    ) -> Dict[str, str]:
        """
        Assign roles with optional pre-selected judge

        Args:
            assessments: Self-assessments from all models
            problem_text: The problem text
            model_names: All available models
            selected_judge: Pre-selected judge (from collaborative voting)

        Returns:
            Role assignments
        """
        # If judge already selected, assign remaining as solvers
        if selected_judge:
            remaining_models = [m for m in model_names if m != selected_judge]

            # Score remaining models for solver role
            solver_scores = []
            for model in remaining_models:
                if model in assessments:
                    score = assessments[model].confidence_by_role.get('Solver', 0.5)
                else:
                    score = 0.5
                solver_scores.append((model, score))

            # Sort by solver score
            solver_scores.sort(key=lambda x: x[1], reverse=True)

            return {
                'judge': selected_judge,
                'solver_1': solver_scores[0][0] if len(solver_scores) > 0 else None,
                'solver_2': solver_scores[1][0] if len(solver_scores) > 1 else None,
                'solver_3': solver_scores[2][0] if len(solver_scores) > 2 else None
            }

        # Original algorithmic assignment (fallback)
        return self._assign_roles_hungarian(
            self._compute_model_scores(assessments, problem_text)
        )

    def _compute_model_scores(
            self,
            assessments: Dict[str, SelfAssessment],
            problem_text: str
    ) -> Dict[str, Dict[str, float]]:
        """Compute scores for each model for each role"""
        problem_needs = self.analyze_problem_type(problem_text)
        model_scores = {}

        for model_name, assessment in assessments.items():
            # Base score from self-confidence
            solver_score = assessment.confidence_by_role.get('Solver', 0.5)
            judge_score = assessment.confidence_by_role.get('Judge', 0.5)

            # Adjust based on problem needs
            reasoning_lower = assessment.reasoning.lower()

            if problem_needs.get('mathematical', 0) > 0.5:
                if any(word in reasoning_lower for word in ['math', 'calculate', 'formula']):
                    solver_score *= 1.2

            if problem_needs.get('analytical', 0) > 0.5 or problem_needs.get('critical', 0) > 0.5:
                if any(word in reasoning_lower for word in ['analyze', 'evaluate', 'critical']):
                    judge_score *= 1.2

            model_scores[model_name] = {
                'Solver': min(solver_score, 1.0),
                'Judge': min(judge_score, 1.0),
                'raw_confidence': assessment.confidence_by_role
            }

        return model_scores

    def _assign_roles_hungarian(self, model_scores: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Simplified Hungarian assignment (original fallback method)"""
        models = list(model_scores.keys())

        # Pick judge - highest judge score
        judge_scores = [(model, model_scores[model]['Judge']) for model in models]
        judge_scores.sort(key=lambda x: x[1], reverse=True)
        judge_model = judge_scores[0][0]

        # Remaining models become solvers
        remaining_models = [m for m in models if m != judge_model]

        # Sort solvers by solver score
        solver_scores = [(model, model_scores[model]['Solver']) for model in remaining_models]
        solver_scores.sort(key=lambda x: x[1], reverse=True)

        assignment = {
            'judge': judge_model,
            'solver_1': solver_scores[0][0] if len(solver_scores) > 0 else None,
            'solver_2': solver_scores[1][0] if len(solver_scores) > 1 else None,
            'solver_3': solver_scores[2][0] if len(solver_scores) > 2 else None
        }

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


# Example usage
if __name__ == "__main__":
    assigner = RoleAssigner()

    # Mock assessments
    mock_assessments = {
        "llama": SelfAssessment(
            role_preferences=["Solver", "Judge"],
            confidence_by_role={"Solver": 0.85, "Judge": 0.70},
            reasoning="I'm strong at mathematical reasoning",
            strengths_for_this_problem=["math", "logic"],
            weaknesses_for_this_problem=["overconfidence"],
            raw_response={}
        ),
        "gemini": SelfAssessment(
            role_preferences=["Judge", "Solver"],
            confidence_by_role={"Solver": 0.75, "Judge": 0.80},
            reasoning="I'm good at evaluating solutions",
            strengths_for_this_problem=["analysis", "comparison"],
            weaknesses_for_this_problem=["math details"],
            raw_response={}
        ),
        "mistral": SelfAssessment(
            role_preferences=["Solver", "Judge"],
            confidence_by_role={"Solver": 0.80, "Judge": 0.65},
            reasoning="Strong at step-by-step solutions",
            strengths_for_this_problem=["logical steps"],
            weaknesses_for_this_problem=["edge cases"],
            raw_response={}
        ),
        "groq": SelfAssessment(
            role_preferences=["Solver", "Judge"],
            confidence_by_role={"Solver": 0.78, "Judge": 0.72},
            reasoning="Balanced capabilities",
            strengths_for_this_problem=["reasoning"],
            weaknesses_for_this_problem=["complex math"],
            raw_response={}
        )
    }

    # Mock votes
    mock_votes = {
        "llama": JudgeVote("llama", {"gemini": 0.9, "mistral": 0.6, "groq": 0.7}, "Gemini best at judging"),
        "gemini": JudgeVote("gemini", {"llama": 0.7, "mistral": 0.5, "groq": 0.6}, "Llama is good solver"),
        "mistral": JudgeVote("mistral", {"gemini": 0.85, "llama": 0.7, "groq": 0.65}, "Gemini for judge"),
        "groq": JudgeVote("groq", {"gemini": 0.88, "llama": 0.72, "mistral": 0.60}, "Gemini analytical")
    }

    models = ["llama", "gemini", "mistral", "groq"]
    problem = "Solve this math problem..."

    # Collaborative judge selection
    selected_judge = assigner.collaborative_judge_selection(mock_assessments, models, mock_votes)

    # Final role assignment
    assignments = assigner.algorithmic_assignment(mock_assessments, problem, models, selected_judge)

    print("\nFinal Role Assignments:")
    for role, model in assignments.items():
        print(f"  {role}: {model}")