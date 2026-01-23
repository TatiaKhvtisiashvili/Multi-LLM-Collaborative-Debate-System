"""
Experiment Runner - Batch runs all experiments and baselines
Person B Responsibility
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import argparse
from tqdm.asyncio import tqdm_asyncio

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset_manager import DatasetManager, Problem
from src.debate_orchestrator import DebateOrchestrator, run_debates_batch
from src.evaluation_metrics import MetricsCalculator, BaselineExperiment
from src.utils import setup_logging, format_timestamp


class ExperimentRunner:
    """Main experiment runner for the debate system"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.results_dir = Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        setup_logging(log_file=self.log_file)

        self.dataset_manager = DatasetManager()

    async def run_full_experiment(self, problem_limit: int = None):
        """Run full experiment: debate system + baselines"""
        print("\n" + "=" * 60)
        print("MULTI-LLM DEBATE SYSTEM EXPERIMENT")
        print("=" * 60 + "\n")

        start_time = datetime.now()

        # Step 1: Load or create dataset
        print("[1/5] Loading dataset...")
        problems = self.dataset_manager.load_dataset()

        if not problems:
            print("Dataset not found. Creating new dataset...")
            problems = await self._create_dataset()

        if problem_limit:
            problems = problems[:problem_limit]
            print(f"Limited to {problem_limit} problems")

        print(f"Loaded {len(problems)} problems")

        # Step 2: Run debate system
        print("\n[2/5] Running debate system...")
        debate_results = await self._run_debate_system(problems)

        # Step 3: Run baselines
        print("\n[3/5] Running baseline experiments...")
        baseline_results = await self._run_baselines(problems)

        # Step 4: Calculate metrics
        print("\n[4/5] Calculating metrics...")
        metrics_report = await self._calculate_metrics(debate_results, baseline_results)

        # Step 5: Generate report
        print("\n[5/5] Generating final report...")
        final_report = self._generate_final_report(
            debate_results, baseline_results, metrics_report, start_time
        )

        # Display summary
        self._display_summary(final_report)

        return final_report

    async def _create_dataset(self) -> List[Problem]:
        """Create dataset if it doesn't exist"""
        # Person A: curated problems
        curated = self.dataset_manager.create_curated_problems()

        # Person B: generated problems (simulated here)
        generated = self.dataset_manager.generate_llm_problems(10)

        all_problems = curated + generated

        if self.dataset_manager.validate_dataset(all_problems):
            self.dataset_manager.save_dataset(all_problems)
            print(f"Created dataset with {len(all_problems)} problems")
        else:
            print("Warning: Dataset validation failed")

        return all_problems

    async def _run_debate_system(self, problems: List[Problem]) -> List[Dict[str, Any]]:
        """Run the full debate system on all problems"""
        # Convert Problem objects to dicts
        problem_dicts = []
        for problem in problems:
            problem_dict = {
                'id': problem.id,
                'category': problem.category,
                'problem': problem.problem,
                'ground_truth_answer': problem.ground_truth_answer,
                'ground_truth_reasoning': problem.ground_truth_reasoning
            }
            problem_dicts.append(problem_dict)

        print(f"Running debate system on {len(problem_dicts)} problems...")

        # Run with concurrency control (2 at a time to avoid rate limits)
        debate_results = await run_debates_batch(
            problem_dicts,
            config_path=self.config_path,
            max_concurrent=2
        )

        # Save individual results
        for result in debate_results:
            if hasattr(result, 'problem_id'):
                filename = f"debate_{result.problem_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                filepath = self.results_dir / filename

                # Convert to dict if needed
                if hasattr(result, '__dict__'):
                    result_dict = result.__dict__
                else:
                    result_dict = result

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(debate_results)} debate results")
        return debate_results

    async def _run_baselines(self, problems: List[Problem]) -> Dict[str, List[bool]]:
        """Run baseline experiments"""
        baseline_experiment = BaselineExperiment(self.config_path)
        problem_dicts = [p.__dict__ for p in problems]

        print("Running single-model baseline...")
        single_model_results = await baseline_experiment.run_single_model_baseline(
            problem_dicts, model_key="llama_3_3_70b"
        )

        print("Running voting baseline...")
        voting_results = await baseline_experiment.run_voting_baseline(
            problem_dicts, model_key="llama_3_3_70b", n_votes=3
        )

        # Save baseline results
        baseline_data = {
            'single_model': single_model_results,
            'voting': voting_results,
            'timestamp': format_timestamp(datetime.now()),
            'total_problems': len(problems)
        }

        baseline_file = self.results_dir / f"baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(baseline_file, 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, indent=2)

        print(f"Saved baseline results to {baseline_file}")

        return {
            'single_model': single_model_results,
            'voting': voting_results
        }

    async def _calculate_metrics(
            self,
            debate_results: List[Dict[str, Any]],
            baseline_results: Dict[str, List[bool]]
    ) -> Dict[str, Any]:
        """Calculate all metrics"""
        # First, ensure all debate results are saved
        calculator = MetricsCalculator(str(self.results_dir))

        # Load results from files
        calculator.load_results()

        # Calculate problem metrics
        problem_metrics = calculator.calculate_problem_metrics()

        # Calculate system metrics
        system_metrics = calculator.calculate_system_metrics()

        # Compare with baselines
        comparison = calculator.compare_with_baselines(baseline_results)

        # Generate detailed report
        detailed_report = calculator.generate_detailed_report(
            output_path=str(self.results_dir / "detailed_evaluation_report.json")
        )

        metrics_report = {
            'system_metrics': system_metrics.__dict__,
            'comparison_with_baselines': comparison,
            'detailed_report_path': str(self.results_dir / "detailed_evaluation_report.json"),
            'timestamp': format_timestamp(datetime.now())
        }

        # Save metrics report
        metrics_file = self.results_dir / f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_report, f, indent=2, default=str)

        return metrics_report

    def _generate_final_report(
            self,
            debate_results: List[Dict[str, Any]],
            baseline_results: Dict[str, List[bool]],
            metrics_report: Dict[str, Any],
            start_time: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Count API calls and tokens from debate orchestrator stats
        total_api_calls = 0
        total_tokens = 0

        for result in debate_results:
            if isinstance(result, dict) and 'model_usage' in result:
                # Sum up usage from all results
                pass  # In real implementation, would sum up

        final_report = {
            'experiment_summary': {
                'start_time': format_timestamp(start_time),
                'end_time': format_timestamp(end_time),
                'total_duration_seconds': total_duration,
                'total_problems': len(debate_results),
                'total_api_calls': total_api_calls,
                'total_tokens_used': total_tokens
            },
            'system_performance': metrics_report.get('system_metrics', {}),
            'baseline_comparison': metrics_report.get('comparison_with_baselines', {}),
            'key_findings': self._extract_key_findings(metrics_report),
            'recommendations': self._generate_recommendations(metrics_report),
            'files_generated': self._list_generated_files()
        }

        # Save final report
        report_file = self.results_dir / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, default=str)

        # Also save as markdown for easy reading
        self._save_markdown_report(final_report, report_file.with_suffix('.md'))

        print(f"\nFinal report saved to: {report_file}")
        return final_report

    def _extract_key_findings(self, metrics_report: Dict[str, Any]) -> List[str]:
        """Extract key findings from metrics"""
        findings = []

        system_metrics = metrics_report.get('system_metrics', {})
        comparison = metrics_report.get('comparison_with_baselines', {})

        # Accuracy findings
        sys_acc = system_metrics.get('overall_accuracy', 0)
        single_acc = comparison.get('baselines', {}).get('single_model', {}).get('accuracy', 0)

        if sys_acc > single_acc:
            improvement = ((sys_acc - single_acc) / single_acc * 100) if single_acc > 0 else 0
            findings.append(f"Debate system achieved {sys_acc:.1%} accuracy, "
                            f"improving over single-model baseline ({single_acc:.1%}) by {improvement:.1f}%")
        else:
            findings.append(f"Debate system accuracy ({sys_acc:.1%}) "
                            f"similar to single-model baseline ({single_acc:.1%})")

        # Refinement findings
        improvement_rate = system_metrics.get('improvement_rate', 0)
        if improvement_rate > 0.1:  # More than 10% improvement
            findings.append(f"Peer review led to solution refinement in {improvement_rate:.1%} of cases")

        # Judge performance
        judge_acc = system_metrics.get('judge_accuracy', 0)
        if judge_acc > 0.7:
            findings.append(
                f"Final judge correctly identified best solution in {judge_acc:.1%} of cases with solver disagreements")

        # Consensus findings
        consensus_rate = system_metrics.get('consensus_rate', 0)
        findings.append(f"Solvers reached initial consensus in {consensus_rate:.1%} of problems")

        return findings

    def _generate_recommendations(self, metrics_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []
        system_metrics = metrics_report.get('system_metrics', {})

        # Based on error analysis
        error_types = system_metrics.get('common_error_types', [])
        if error_types:
            most_common = error_types[0][0] if error_types else None
            if most_common == 'parse_error':
                recommendations.append("Improve JSON parsing robustness or use more structured output formats")

        # Based on confidence analysis
        conf_corr = system_metrics.get('confidence_correlation', 0)
        if abs(conf_corr) < 0.3:
            recommendations.append(
                "Implement better confidence calibration - current confidence scores don't correlate well with correctness")

        # Based on processing time
        avg_time = system_metrics.get('avg_processing_time', 0)
        if avg_time > 60:  # More than 60 seconds per problem
            recommendations.append(f"Optimize system performance - average {avg_time:.1f}s per problem is high")

        # General recommendations
        recommendations.append("Consider adding more diverse models to the solver pool")
        recommendations.append("Implement more sophisticated peer review matching based on problem type")
        recommendations.append("Add human evaluation component for ambiguous cases")

        return recommendations

    def _list_generated_files(self) -> List[str]:
        """List all files generated during experiment"""
        files = []

        # Results directory
        for file_path in self.results_dir.glob("*.json"):
            files.append(str(file_path.relative_to(self.results_dir.parent)))

        # Log directory
        for file_path in self.log_dir.glob("*.log"):
            files.append(str(file_path.relative_to(self.log_dir.parent)))

        return files

    def _save_markdown_report(self, report: Dict[str, Any], output_path: Path):
        """Save report as markdown for easy reading"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Multi-LLM Debate System Experiment Report\n\n")

            f.write("## Experiment Summary\n")
            f.write(f"- **Start Time**: {report['experiment_summary']['start_time']}\n")
            f.write(f"- **End Time**: {report['experiment_summary']['end_time']}\n")
            f.write(f"- **Total Duration**: {report['experiment_summary']['total_duration_seconds']:.1f} seconds\n")
            f.write(f"- **Total Problems**: {report['experiment_summary']['total_problems']}\n")
            f.write(f"- **Total API Calls**: {report['experiment_summary']['total_api_calls']}\n")
            f.write(f"- **Total Tokens Used**: {report['experiment_summary']['total_tokens_used']}\n\n")

            f.write("## System Performance\n")
            metrics = report['system_performance']
            f.write(f"- **Overall Accuracy**: {metrics.get('overall_accuracy', 0):.2%}\n")
            f.write(f"- **Improvement Rate**: {metrics.get('improvement_rate', 0):.2%}\n")
            f.write(f"- **Consensus Rate**: {metrics.get('consensus_rate', 0):.2%}\n")
            f.write(f"- **Judge Accuracy**: {metrics.get('judge_accuracy', 0):.2%}\n")
            f.write(f"- **Average Processing Time**: {metrics.get('avg_processing_time', 0):.2f}s\n\n")

            f.write("## Comparison with Baselines\n")
            comparison = report['baseline_comparison']
            f.write(f"- **System Accuracy**: {comparison.get('system_accuracy', 0):.2%}\n")

            for baseline_name, baseline_data in comparison.get('baselines', {}).items():
                f.write(f"- **{baseline_name.title()} Baseline**: {baseline_data.get('accuracy', 0):.2%}\n")
                f.write(f"  - Improvement: {baseline_data.get('improvement_over_baseline', 0):.2%}\n")
                f.write(f"  - Relative Improvement: {baseline_data.get('relative_improvement', 0):.2%}\n")

            f.write("\n## Key Findings\n")
            for finding in report.get('key_findings', []):
                f.write(f"- {finding}\n")

            f.write("\n## Recommendations\n")
            for recommendation in report.get('recommendations', []):
                f.write(f"- {recommendation}\n")

            f.write("\n## Generated Files\n")
            for file_path in report.get('files_generated', []):
                f.write(f"- `{file_path}`\n")

    def _display_summary(self, report: Dict[str, Any]):
        """Display summary to console"""
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE - SUMMARY")
        print("=" * 60)

        print(f"\nPerformance Metrics:")
        print(f"   Overall Accuracy:     {report['system_performance'].get('overall_accuracy', 0):.2%}")
        print(f"   Improvement Rate:     {report['system_performance'].get('improvement_rate', 0):.2%}")
        print(f"   Consensus Rate:       {report['system_performance'].get('consensus_rate', 0):.2%}")
        print(f"   Judge Accuracy:       {report['system_performance'].get('judge_accuracy', 0):.2%}")

        print(f"\nProcessing:")
        print(f"   Total Problems:       {report['experiment_summary']['total_problems']}")
        print(f"   Total Duration:       {report['experiment_summary']['total_duration_seconds']:.1f}s")
        print(f"   Avg Time per Problem: {report['system_performance'].get('avg_processing_time', 0):.1f}s")

        print(f"\nComparison with Baselines:")
        comparison = report['baseline_comparison']
        for baseline_name, baseline_data in comparison.get('baselines', {}).items():
            improvement = baseline_data.get('improvement_over_baseline', 0)
            arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
            print(f"   {baseline_name}: {baseline_data.get('accuracy', 0):.2%} "
                  f"({arrow} {improvement:+.2%})")

        print(f"\nKey Findings:")
        for finding in report.get('key_findings', [])[:3]:  # Show top 3
            print(f"   • {finding}")

        print(f"\nReports saved to: data/results/")
        print("=" * 60)


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run Multi-LLM Debate System Experiment')
    parser.add_argument('--limit', type=int, help='Limit number of problems to process')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--skip-baselines', action='store_true', help='Skip baseline experiments')

    args = parser.parse_args()

    print("🔧 Initializing experiment runner...")
    runner = ExperimentRunner(config_path=args.config)

    try:
        await runner.run_full_experiment(problem_limit=args.limit)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nError running experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nExperiment completed successfully!")
    return 0


if __name__ == "__main__":
    # Run async main
    import asyncio

    exit_code = asyncio.run(main())
    sys.exit(exit_code)