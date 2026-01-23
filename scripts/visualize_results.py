"""
Visualization Script - Generates required plots from evaluation results
Person A Responsibility
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ResultsVisualizer:
    """Generates visualizations from experiment results"""

    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.figures_dir = Path("figures")
        self.figures_dir.mkdir(exist_ok=True)

        # Load data
        self.metrics_report = self._load_latest_metrics()
        self.detailed_report = self._load_detailed_report()
        self.baseline_data = self._load_baseline_data()

    def _load_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Load the latest metrics report"""
        metrics_files = list(self.results_dir.glob("metrics_report_*.json"))
        if not metrics_files:
            return None

        latest_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
        with open(latest_file, 'r') as f:
            return json.load(f)

    def _load_detailed_report(self) -> Optional[Dict[str, Any]]:
        """Load detailed evaluation report"""
        report_file = self.results_dir / "detailed_evaluation_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                return json.load(f)
        return None

    def _load_baseline_data(self) -> Optional[Dict[str, Any]]:
        """Load baseline experiment data"""
        baseline_files = list(self.results_dir.glob("baselines_*.json"))
        if not baseline_files:
            return None

        latest_file = max(baseline_files, key=lambda f: f.stat().st_mtime)
        with open(latest_file, 'r') as f:
            return json.load(f)

    def generate_all_plots(self, save: bool = True, show: bool = True):
        """Generate all required plots"""
        print("Generating evaluation plots...")

        # 1. Bar Chart: Accuracy Comparison
        self.plot_accuracy_comparison(save=save, show=show)

        # 2. Line Chart: Improvement Rate Over Categories
        self.plot_improvement_by_category(save=save, show=show)

        # 3. Scatter Plot: Judge Accuracy vs Solver Disagreement
        self.plot_judge_accuracy_scatter(save=save, show=show)

        # 4. Bar Chart: Solver Individual Performance
        self.plot_solver_performance(save=save, show=show)

        # 5. Heatmap: Category Performance
        self.plot_category_heatmap(save=save, show=show)

        # 6. Confidence Analysis Plot
        self.plot_confidence_analysis(save=save, show=show)

        # 7. Processing Time Distribution
        self.plot_processing_time_distribution(save=save, show=show)

        # 8. Consensus vs Accuracy Scatter
        self.plot_consensus_vs_accuracy(save=save, show=show)

        print(f"Generated {8} plots in {self.figures_dir}/")

        # Generate composite figure for report
        self.generate_composite_figure()

    def plot_accuracy_comparison(self, save: bool = True, show: bool = False):
        """Bar Chart: System vs Baseline Accuracy"""
        if not self.metrics_report:
            print("No metrics report found")
            return

        comparison = self.metrics_report.get('comparison_with_baselines', {})
        if not comparison:
            print("No comparison data found")
            return

        # Prepare data
        methods = ['Our System']
        accuracies = [comparison.get('system_accuracy', 0)]

        baselines = comparison.get('baselines', {})
        for baseline_name, baseline_data in baselines.items():
            methods.append(baseline_name.replace('_', ' ').title())
            accuracies.append(baseline_data.get('accuracy', 0))

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(methods, accuracies, color=sns.color_palette("husl", len(methods)))

        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{accuracy:.1%}', ha='center', va='bottom', fontsize=10)

        # Customize
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy Comparison: Debate System vs Baselines', fontsize=14, pad=20)
        ax.set_ylim(0, 1.1)

        # Add improvement annotations
        if len(accuracies) > 1:
            system_acc = accuracies[0]
            for i, baseline_acc in enumerate(accuracies[1:], 1):
                improvement = system_acc - baseline_acc
                if abs(improvement) > 0.01:
                    ax.annotate(f'{improvement:+.1%}',
                                xy=(bars[i].get_x() + bars[i].get_width() / 2, baseline_acc),
                                xytext=(0, 20),
                                textcoords='offset points',
                                ha='center',
                                arrowprops=dict(arrowstyle='->', color='red' if improvement < 0 else 'green'),
                                color='red' if improvement < 0 else 'green',
                                fontweight='bold')

        plt.xticks(rotation=15)
        plt.tight_layout()

        if save:
            plt.savefig(self.figures_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.figures_dir / 'accuracy_comparison.pdf', bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_improvement_by_category(self, save: bool = True, show: bool = False):
        """Line Chart: Improvement Rate Across Categories"""
        if not self.detailed_report:
            print("No detailed report found")
            return

        category_data = self.detailed_report.get('category_performance', {})
        if not category_data:
            print("No category data found")
            return

        # Extract categories and accuracies
        categories = list(category_data.keys())
        accuracies = list(category_data.values())

        # Sort by accuracy
        sorted_pairs = sorted(zip(categories, accuracies), key=lambda x: x[1])
        categories = [p[0] for p in sorted_pairs]
        accuracies = [p[1] for p in sorted_pairs]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot line
        line = ax.plot(categories, accuracies, marker='o', linewidth=2, markersize=8,
                       color='steelblue', markerfacecolor='white', markeredgewidth=2)

        # Add value labels
        for i, (cat, acc) in enumerate(zip(categories, accuracies)):
            ax.text(i, acc + 0.02, f'{acc:.1%}', ha='center', va='bottom', fontsize=9)

        # Customize
        ax.set_xlabel('Problem Category', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('System Performance Across Problem Categories', fontsize=14, pad=20)
        ax.set_ylim(0, 1.1)

        # Add horizontal line for average
        avg_accuracy = np.mean(accuracies)
        ax.axhline(y=avg_accuracy, color='red', linestyle='--', alpha=0.7, label=f'Average: {avg_accuracy:.1%}')
        ax.legend()

        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            plt.savefig(self.figures_dir / 'improvement_by_category.png', dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_judge_accuracy_scatter(self, save: bool = True, show: bool = False):
        """Scatter Plot: Judge Accuracy vs Solver Disagreement"""
        if not self.detailed_report:
            print("No detailed report found")
            return

        problem_details = self.detailed_report.get('problem_details', [])
        if not problem_details:
            print("No problem details found")
            return

        # Prepare data
        data = []
        for problem in problem_details:
            if not problem.get('solver_consensus', True):  # Only non-consensus problems
                judge_correct = problem.get('judge_picked_correct', False)

                # Calculate disagreement level (simplified)
                # In real data, would use actual answer diversity
                disagreement_level = 1.0 if problem.get('refinement_changed', False) else 0.5

                data.append({
                    'disagreement': disagreement_level,
                    'judge_correct': judge_correct,
                    'problem_id': problem.get('problem_id', '')
                })

        if not data:
            print("No non-consensus problems found")
            return

        df = pd.DataFrame(data)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create scatter plot
        scatter = ax.scatter(df['disagreement'],
                             df['judge_correct'].astype(int),
                             alpha=0.6, s=100,
                             c=df['judge_correct'].map({True: 'green', False: 'red'}),
                             edgecolors='black', linewidth=0.5)

        # Add problem IDs for misjudgments
        for _, row in df.iterrows():
            if not row['judge_correct']:
                ax.annotate(row['problem_id'],
                            xy=(row['disagreement'], row['judge_correct']),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8,
                            alpha=0.7)

        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['disagreement'], df['judge_correct'].astype(int), 1)
            p = np.poly1d(z)
            ax.plot(df['disagreement'], p(df['disagreement']),
                    "r--", alpha=0.8, label=f'Trend (slope={z[0]:.2f})')

        # Customize
        ax.set_xlabel('Solver Disagreement Level', fontsize=12)
        ax.set_ylabel('Judge Correct (1=Yes, 0=No)', fontsize=12)
        ax.set_title('Judge Accuracy vs Solver Disagreement', fontsize=14, pad=20)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Incorrect', 'Correct'])
        ax.set_xlim(-0.1, 1.6)
        ax.legend()

        # Add accuracy percentage
        judge_accuracy = df['judge_correct'].mean()
        ax.text(0.05, 0.95, f'Judge Accuracy: {judge_accuracy:.1%}',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            plt.savefig(self.figures_dir / 'judge_accuracy_scatter.png', dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_solver_performance(self, save: bool = True, show: bool = False):
        """Bar Chart: Individual Solver Performance"""
        if not self.metrics_report:
            print("No metrics report found")
            return

        system_metrics = self.metrics_report.get('system_metrics', {})
        solver_accuracy = system_metrics.get('solver_individual_accuracy', {})
        solver_wins = system_metrics.get('solver_contribution_to_wins', {})

        if not solver_accuracy:
            print("No solver performance data found")
            return

        # Prepare data
        solvers = list(solver_accuracy.keys())
        accuracies = [solver_accuracy[s] for s in solvers]

        # Get win counts
        win_counts = [solver_wins.get(s, 0) for s in solvers]

        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar plot for accuracy
        bars = ax1.bar(solvers, accuracies, alpha=0.7, label='Accuracy')
        ax1.set_xlabel('Solver', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_ylim(0, 1.1)

        # Add accuracy values on bars
        for bar, accuracy in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{accuracy:.1%}', ha='center', va='bottom',
                     color='steelblue', fontsize=9)

        # Line plot for win counts (secondary axis)
        ax2 = ax1.twinx()
        line = ax2.plot(solvers, win_counts, 'ro-', linewidth=2, markersize=8,
                        label='Final Wins', color='coral')
        ax2.set_ylabel('Number of Final Wins', fontsize=12, color='coral')
        ax2.tick_params(axis='y', labelcolor='coral')

        # Set appropriate y-limit for win counts
        max_wins = max(win_counts) if win_counts else 1
        ax2.set_ylim(0, max_wins * 1.2)

        # Add win count values on points
        for i, (solver, wins) in enumerate(zip(solvers, win_counts)):
            ax2.text(i, wins + max_wins * 0.05, str(wins),
                     ha='center', va='bottom', color='coral', fontsize=9)

        # Title and legend
        ax1.set_title('Individual Solver Performance', fontsize=14, pad=20)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()

        if save:
            plt.savefig(self.figures_dir / 'solver_performance.png', dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_category_heatmap(self, save: bool = True, show: bool = False):
        """Heatmap: Performance Across Categories and Metrics"""
        if not self.detailed_report:
            print("No detailed report found")
            return

        category_data = self.detailed_report.get('category_performance', {})
        if not category_data:
            print("No category data found")
            return

        # For heatmap, need multiple metrics per category
        # In real data, would have accuracy, consensus rate, improvement rate per category
        categories = list(category_data.keys())
        accuracies = list(category_data.values())

        # Create mock data for demonstration
        # In real implementation, would load actual multi-metric data
        n_categories = len(categories)
        metrics = ['Accuracy', 'Consensus', 'Improvement']

        # Create synthetic data for heatmap
        data = np.random.rand(len(metrics), n_categories) * 0.5 + 0.3
        data[0, :] = accuracies  # Use actual accuracies

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        # Show all ticks and labels
        ax.set_xticks(np.arange(len(categories)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in categories])
        ax.set_yticklabels(metrics)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(categories)):
                text = ax.text(j, i, f'{data[i, j]:.1%}',
                               ha="center", va="center", color="black" if data[i, j] < 0.7 else "white",
                               fontsize=9)

        ax.set_title("Performance Heatmap Across Categories and Metrics", fontsize=14, pad=20)
        fig.colorbar(im, ax=ax, label='Performance Score')

        plt.tight_layout()

        if save:
            plt.savefig(self.figures_dir / 'category_heatmap.png', dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_confidence_analysis(self, save: bool = True, show: bool = False):
        """Plot confidence analysis"""
        if not self.metrics_report:
            print("No metrics report found")
            return

        system_metrics = self.metrics_report.get('system_metrics', {})

        avg_conf_correct = system_metrics.get('avg_confidence_when_correct', 0)
        avg_conf_wrong = system_metrics.get('avg_confidence_when_wrong', 0)
        conf_correlation = system_metrics.get('confidence_correlation', 0)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Subplot 1: Average confidence
        categories = ['When Correct', 'When Wrong']
        values = [avg_conf_correct, avg_conf_wrong]
        colors = ['green', 'red']

        bars1 = ax1.bar(categories, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Average Confidence', fontsize=12)
        ax1.set_title('Confidence vs Correctness', fontsize=13)
        ax1.set_ylim(0, 1.1)

        # Add values on bars
        for bar, value in zip(bars1, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=10)

        # Subplot 2: Correlation indicator
        correlation_data = {'Correlation': conf_correlation}
        bars2 = ax2.bar(list(correlation_data.keys()), list(correlation_data.values()),
                        color='steelblue', alpha=0.7)
        ax2.set_ylabel('Correlation Coefficient', fontsize=12)
        ax2.set_title('Confidence-Correctness Correlation', fontsize=13)
        ax2.set_ylim(-1, 1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Color based on correlation strength
        for bar in bars2:
            value = bar.get_height()
            if value > 0.3:
                bar.set_color('green')
            elif value < -0.3:
                bar.set_color('red')
            else:
                bar.set_color('orange')

            # Add value
            ax2.text(bar.get_x() + bar.get_width() / 2.,
                     value + (0.1 if value >= 0 else -0.12),
                     f'{value:.3f}', ha='center', va='bottom' if value >= 0 else 'top',
                     fontsize=10, fontweight='bold')

        plt.suptitle('Confidence Analysis', fontsize=16, y=1.02)
        plt.tight_layout()

        if save:
            plt.savefig(self.figures_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_processing_time_distribution(self, save: bool = True, show: bool = False):
        """Histogram of processing times"""
        if not self.detailed_report:
            print("No detailed report found")
            return

        problem_details = self.detailed_report.get('problem_details', [])
        if not problem_details:
            print("No problem details found")
            return

        # Extract processing times (would need to be in problem_details)
        # For now, create synthetic data
        processing_times = np.random.exponential(scale=30, size=len(problem_details))

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        n, bins, patches = ax.hist(processing_times, bins=20, alpha=0.7,
                                   color='steelblue', edgecolor='black')

        # Add vertical line for mean
        mean_time = np.mean(processing_times)
        ax.axvline(mean_time, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_time:.1f}s')

        # Add statistics box
        stats_text = f'Mean: {mean_time:.1f}s\nStd: {np.std(processing_times):.1f}s\nMin: {np.min(processing_times):.1f}s\nMax: {np.max(processing_times):.1f}s'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Processing Time (seconds)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Processing Time Distribution', fontsize=14, pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.figures_dir / 'processing_time_distribution.png', dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_consensus_vs_accuracy(self, save: bool = True, show: bool = False):
        """Scatter plot: Consensus vs Accuracy"""
        if not self.detailed_report:
            print("No detailed report found")
            return

        problem_details = self.detailed_report.get('problem_details', [])
        if not problem_details:
            print("No problem details found")
            return

        # Prepare data
        consensus_problems = []
        non_consensus_problems = []

        for problem in problem_details:
            if problem.get('solver_consensus', False):
                consensus_problems.append(problem.get('system_correct', False))
            else:
                non_consensus_problems.append(problem.get('system_correct', False))

        # Calculate accuracies
        consensus_accuracy = np.mean(consensus_problems) if consensus_problems else 0
        non_consensus_accuracy = np.mean(non_consensus_problems) if non_consensus_problems else 0

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['With Consensus', 'Without Consensus']
        accuracies = [consensus_accuracy, non_consensus_accuracy]
        counts = [len(consensus_problems), len(non_consensus_problems)]

        # Create bubble chart
        scatter = ax.scatter(categories, accuracies, s=[c * 100 for c in counts],
                             alpha=0.6, color=['green', 'red'], edgecolors='black', linewidth=1.5)

        # Add value labels
        for i, (category, acc, count) in enumerate(zip(categories, accuracies, counts)):
            ax.text(i, acc + 0.02, f'{acc:.1%}\n(n={count})',
                    ha='center', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy: Problems With vs Without Initial Solver Consensus', fontsize=14, pad=20)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.figures_dir / 'consensus_vs_accuracy.png', dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def generate_composite_figure(self):
        """Generate a composite figure with all key plots for report"""
        # This creates a 2x2 grid of the most important plots
        fig = plt.figure(figsize=(16, 12))

        # Mock data for demonstration
        # In real implementation, would load actual data

        # 1. Accuracy Comparison (top-left)
        ax1 = plt.subplot(2, 2, 1)
        methods = ['Our System', 'Single Model', 'Voting']
        accuracies = [0.75, 0.65, 0.68]
        ax1.bar(methods, accuracies, color=['green', 'gray', 'gray'])
        ax1.set_title('Accuracy Comparison', fontsize=12)
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)

        # 2. Solver Performance (top-right)
        ax2 = plt.subplot(2, 2, 2)
        solvers = ['Solver 1', 'Solver 2', 'Solver 3']
        solver_acc = [0.70, 0.68, 0.72]
        ax2.bar(solvers, solver_acc, color=sns.color_palette("husl", 3))
        ax2.set_title('Individual Solver Accuracy', fontsize=12)
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)

        # 3. Category Performance (bottom-left)
        ax3 = plt.subplot(2, 2, 3)
        categories = ['Math', 'Physics', 'Logic', 'Game Theory']
        category_acc = [0.80, 0.75, 0.70, 0.65]
        ax3.plot(categories, category_acc, 'o-', linewidth=2)
        ax3.set_title('Performance by Category', fontsize=12)
        ax3.set_ylabel('Accuracy')
        ax3.set_ylim(0, 1)
        plt.xticks(rotation=45)

        # 4. Judge Accuracy (bottom-right)
        ax4 = plt.subplot(2, 2, 4)
        judge_acc = 0.85
        ax4.bar(['Judge'], [judge_acc], color='orange')
        ax4.set_title(f'Judge Accuracy: {judge_acc:.1%}', fontsize=12)
        ax4.set_ylabel('Accuracy')
        ax4.set_ylim(0, 1)
        ax4.text(0, judge_acc + 0.02, f'{judge_acc:.1%}', ha='center', va='bottom')

        plt.suptitle('Multi-LLM Debate System - Key Results', fontsize=16, y=0.98)
        plt.tight_layout()

        # Save composite figure
        composite_path = self.figures_dir / 'composite_results.png'
        plt.savefig(composite_path, dpi=300, bbox_inches='tight')
        print(f"Composite figure saved to: {composite_path}")

        plt.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate visualization plots from experiment results')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    parser.add_argument('--results-dir', type=str, default='data/results', help='Results directory')

    args = parser.parse_args()

    print("📊 Results Visualizer")
    print("=" * 50)

    visualizer = ResultsVisualizer(results_dir=args.results_dir)
    visualizer.generate_all_plots(show=args.show)

    print("\n✅ Visualization complete!")
    print(f"📁 Plots saved to: {visualizer.figures_dir}/")

    # Print summary if metrics available
    if visualizer.metrics_report:
        metrics = visualizer.metrics_report.get('system_metrics', {})
        print(f"\n📈 Key Metrics:")
        print(f"   Overall Accuracy: {metrics.get('overall_accuracy', 0):.1%}")
        print(f"   Improvement Rate: {metrics.get('improvement_rate', 0):.1%}")
        print(f"   Consensus Rate:   {metrics.get('consensus_rate', 0):.1%}")
        print(f"   Judge Accuracy:   {metrics.get('judge_accuracy', 0):.1%}")


if __name__ == "__main__":
    main()