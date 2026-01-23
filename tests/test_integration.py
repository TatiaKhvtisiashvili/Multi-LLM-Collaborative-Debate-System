## 12. Test Script

### `tests/test_integration.py`
"""
Integration Tests - Tests the complete system with sample problems
"""
import pytest
import asyncio
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.debate_orchestrator import DebateOrchestrator
from src.dataset_manager import DatasetManager
from src.role_assigner import RoleAssigner
from src.parsing_utils import LLMOutputParser


class TestDebateSystem:
    """Integration tests for the debate system"""

    @pytest.fixture
    def sample_problem(self):
        """Sample problem for testing"""
        return {
            "id": "TEST_001",
            "category": "Mathematical/Logical Reasoning",
            "problem": "What is 2 + 2?",
            "ground_truth_answer": "4",
            "ground_truth_reasoning": "Basic arithmetic"
        }

    @pytest.fixture
    def orchestrator(self):
        """Create debate orchestrator"""
        return DebateOrchestrator(config_path="config.yaml")

    def test_role_assigner(self):
        """Test role assignment logic"""
        assigner = RoleAssigner()

        # Test problem analysis
        problem = "Solve for x: 2x + 5 = 13"
        needs = assigner.analyze_problem_type(problem)
        assert 'mathematical' in needs

        # Test parsing
        response = {
            "role_preferences": ["Solver", "Judge"],
            "confidence_by_role": {"Solver": 0.8, "Judge": 0.6},
            "reasoning": "I'm good at math"
        }

        assessment = assigner.parse_self_assessment(
            json.dumps(response), "test_model"
        )
        assert assessment.role_preferences == ["Solver", "Judge"]
        assert assessment.confidence_by_role["Solver"] == 0.8

    def test_parser(self):
        """Test LLM output parsing"""
        parser = LLMOutputParser()

        # Test valid JSON
        valid_json = '{"answer": "42", "confidence": 0.9}'
        result = parser.parse_response(valid_json, "json")
        assert result["parse_success"] == True
        assert result["answer"] == "42"

        # Test with markdown
        markdown_json = '```json\n{"answer": "42"}\n```'
        result = parser.parse_response(markdown_json, "json")
        assert result["parse_success"] == True

        # Test invalid JSON recovery
        bad_json = '{answer: "42"}'  # Missing quotes
        result = parser.parse_response(bad_json, "json")
        # Should either parse successfully or have error info
        assert "parse_success" in result

    def test_dataset_manager(self):
        """Test dataset creation and validation"""
        manager = DatasetManager("test_problems.jsonl")

        # Test problem creation
        problem = manager.create_curated_problems()[0]
        assert hasattr(problem, 'id')
        assert hasattr(problem, 'problem')
        assert hasattr(problem, 'ground_truth_answer')

        # Test validation
        problems = [problem]
        assert manager.validate_dataset(problems) == True

        # Cleanup
        Path("test_problems.jsonl").unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_debate_pipeline_mock(self, sample_problem):
        """Test the complete debate pipeline with mock clients"""
        # This would use mock clients in a real test
        # For now, just test the structure

        orchestrator = DebateOrchestrator()

        # Test that orchestrator can be created
        assert hasattr(orchestrator, 'run_debate')
        assert hasattr(orchestrator, 'save_result')

        # Test statistics tracking
        stats = orchestrator.get_statistics()
        assert 'problems_processed' in stats
        assert 'total_api_calls' in stats

    def test_metrics_calculation(self):
        """Test metrics calculation with sample data"""
        from src.evaluation_metrics import MetricsCalculator, ProblemMetrics

        # Create sample metrics
        metrics = [
            ProblemMetrics(
                problem_id="TEST_001",
                category="Math",
                system_correct=True,
                system_answer="4",
                ground_truth="4",
                solver_answers=["4", "4", "4"],
                refined_answers=["4", "4", "4"],
                final_judgement={"winner": "solver_1"},
                solver_consensus=True,
                refinement_changed=False,
                judge_picked_correct=True,
                processing_time=30.0
            ),
            ProblemMetrics(
                problem_id="TEST_002",
                category="Physics",
                system_correct=False,
                system_answer="5",
                ground_truth="4",
                solver_answers=["4", "5", "4"],
                refined_answers=["4", "5", "4"],
                final_judgement={"winner": "solver_2"},
                solver_consensus=False,
                refinement_changed=False,
                judge_picked_correct=False,
                processing_time=40.0
            )
        ]

        # Test calculator (simplified)
        calculator = MetricsCalculator()
        calculator.problems_metrics = metrics

        system_metrics = calculator.calculate_system_metrics()

        assert system_metrics.overall_accuracy == 0.5  # 1 out of 2 correct
        assert system_metrics.consensus_rate == 0.5  # 1 out of 2 consensus
        assert system_metrics.avg_processing_time == 35.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])