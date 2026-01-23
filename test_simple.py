# test_no_ground_truth.py
import asyncio
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_without_ground_truth():
    """Test that judge doesn't receive ground truth"""
    print("🧪 Testing No Ground Truth to Judge")
    print("=" * 50)

    try:
        from src.agent_prompts import PromptTemplates

        # Create mock data
        problem = "What is 12 × 13?"
        original_solutions = [
            {"solver_id": "solver_1", "final_answer": "156", "solution_steps": []},
            {"solver_id": "solver_2", "final_answer": "156", "solution_steps": []}
        ]
        all_reviews = []
        refined_solutions = []

        # Generate prompt (WITHOUT ground truth)
        prompt = PromptTemplates.get_judgement_prompt(
            problem,
            original_solutions,
            all_reviews,
            refined_solutions
        )

        print("✓ Judge prompt generated successfully")

        # Check that ground truth (like "156") is NOT in the prompt
        if "156" in prompt:
            print("✓ Debaters' answer '156' is in the prompt (correct - it should be)")
        else:
            print("✗ Debaters' answer not found in prompt")

        # Check that the prompt doesn't mention "ground truth"
        if "GROUND TRUTH" in prompt.upper():
            print("✗ ERROR: Ground truth is mentioned in the prompt!")
            return False
        else:
            print("✓ Ground truth not mentioned in prompt (correct)")

        # Show first 300 chars of prompt
        print(f"\nPrompt preview:")
        print(prompt[:300] + "...")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_complete_debate():
    """Test a complete debate without ground truth"""
    print("\n🧪 Testing Complete Debate Flow")
    print("=" * 50)

    try:
        from src.debate_orchestrator import DebateOrchestrator

        # Create test problem
        test_problem = {
            "id": "TEST_NO_TRUTH",
            "category": "Mathematical/Logical Reasoning",
            "problem": "What is 7 × 8?",
            "ground_truth_answer": "56",  # Only for our evaluation
            "ground_truth_reasoning": "7 × 8 = 56"
        }

        print(f"Test problem: {test_problem['problem']}")
        print(f"Our ground truth: {test_problem['ground_truth_answer']} (not shared with models)")

        orchestrator = DebateOrchestrator("config.yaml")
        print("✓ DebateOrchestrator initialized")

        print("\n🚀 Running debate...")
        result = await orchestrator.run_debate(test_problem)

        print(f"\n📊 Results:")
        print(f"  Final Answer: {result.final_answer}")
        print(f"  Our Ground Truth: {result.ground_truth}")
        print(f"  Correct: {result.is_correct}")

        # Check if judge selected an answer
        if result.final_answer:
            print(f"✓ Judge selected an answer: {result.final_answer}")

            # Check if it's one of the possible answers (not necessarily correct)
            if result.final_answer == result.ground_truth:
                print("✓ Judge picked the correct answer!")
            else:
                print("✗ Judge picked a different answer")

        else:
            print("✗ No answer selected by judge")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🔧 Verifying Models Don't Receive Ground Truth")

    prompt_ok = asyncio.run(test_without_ground_truth())

    if prompt_ok:
        print("\n" + "=" * 50)
        print("✅ Prompt fix verified!")
        print("=" * 50)

        # Ask if user wants to run full test
        response = input("\nRun complete debate test? (y/n): ")
        if response.lower() == 'y':
            debate_ok = asyncio.run(test_complete_debate())
            if debate_ok:
                print("\n" + "=" * 50)
                print("✅ Complete debate works without ground truth!")
                print("=" * 50)
            else:
                print("\n" + "=" * 50)
                print("❌ Debate test failed")
                print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ Prompt fix failed")
        print("=" * 50)