"""
Benchmarking System for RL-LLM Performance Evaluation

This module provides comprehensive benchmarking capabilities for evaluating
RL agents, environments, and training algorithms with standardized metrics
and comparison tools.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import logging
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import psutil
import torch
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    benchmark_name: str
    agent_name: str
    environment_name: str
    metrics: Dict[str, float]
    execution_time: float
    timestamp: str
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of related benchmarks."""
    name: str
    description: str
    benchmarks: List[str]
    environments: List[str]
    agents: List[str]
    num_runs: int = 5
    max_episodes: int = 100
    timeout: float = 3600.0  # 1 hour


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.name = self.__class__.__name__
        self.results = []
        
    @abstractmethod
    def setup(self, agent, environment):
        """Setup benchmark with agent and environment."""
        pass
    
    @abstractmethod
    def run_benchmark(self, agent, environment) -> Dict[str, float]:
        """Run benchmark and return metrics."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup benchmark resources."""
        pass
    
    def execute(self, agent, environment) -> BenchmarkResult:
        """Execute complete benchmark."""
        start_time = time.time()
        
        try:
            self.setup(agent, environment)
            metrics = self.run_benchmark(agent, environment)
            execution_time = time.time() - start_time
            
            result = BenchmarkResult(
                benchmark_name=self.name,
                agent_name=getattr(agent, 'name', agent.__class__.__name__),
                environment_name=getattr(environment, 'spec', {}).get('id', environment.__class__.__name__),
                metrics=metrics,
                execution_time=execution_time,
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                config=self.config
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Benchmark {self.name} failed: {e}")
            raise
        finally:
            self.cleanup()


class PerformanceBenchmark(BaseBenchmark):
    """Benchmark for agent performance evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_episodes = config.get('num_episodes', 50)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 1000)
        self.warmup_episodes = config.get('warmup_episodes', 5)
        
    def setup(self, agent, environment):
        """Setup performance benchmark."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.success_rates = []
        
    def run_benchmark(self, agent, environment) -> Dict[str, float]:
        """Run performance benchmark."""
        logger.info(f"Running performance benchmark for {self.num_episodes} episodes")
        
        # Warmup episodes
        for _ in range(self.warmup_episodes):
            self._run_episode(agent, environment, warmup=True)
        
        # Benchmark episodes
        for episode in range(self.num_episodes):
            episode_start = time.time()
            
            obs = environment.reset()
            total_reward = 0.0
            episode_length = 0
            
            for step in range(self.max_steps_per_episode):
                action = agent.act(obs)
                next_obs, reward, done, info = environment.step(action)
                
                total_reward += reward
                episode_length += 1
                obs = next_obs
                
                if done:
                    break
            
            episode_time = time.time() - episode_start
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            self.episode_times.append(episode_time)
            self.success_rates.append(info.get('success', total_reward > 0))
            
            if episode % 10 == 0:
                logger.debug(f"Episode {episode}: Reward={total_reward:.2f}, Length={episode_length}")
        
        # Calculate metrics
        metrics = {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'median_reward': np.median(self.episode_rewards),
            'mean_episode_length': np.mean(self.episode_lengths),
            'std_episode_length': np.std(self.episode_lengths),
            'mean_episode_time': np.mean(self.episode_times),
            'success_rate': np.mean(self.success_rates),
            'total_episodes': self.num_episodes,
            'episodes_per_second': self.num_episodes / sum(self.episode_times)
        }
        
        return metrics
    
    def _run_episode(self, agent, environment, warmup=False):
        """Run a single episode."""
        obs = environment.reset()
        
        for _ in range(self.max_steps_per_episode):
            action = agent.act(obs)
            obs, reward, done, info = environment.step(action)
            
            if done:
                break
    
    def cleanup(self):
        """Cleanup performance benchmark."""
        pass


class EfficiencyBenchmark(BaseBenchmark):
    """Benchmark for computational efficiency."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_samples = config.get('num_samples', 1000)
        self.batch_sizes = config.get('batch_sizes', [1, 8, 32, 128])
        
    def setup(self, agent, environment):
        """Setup efficiency benchmark."""
        self.timing_results = defaultdict(list)
        self.memory_results = defaultdict(list)
        
    def run_benchmark(self, agent, environment) -> Dict[str, float]:
        """Run efficiency benchmark."""
        logger.info("Running efficiency benchmark")
        
        # Get sample observation
        obs = environment.reset()
        
        # Test action generation speed
        action_times = self._benchmark_action_generation(agent, obs)
        
        # Test learning speed (if applicable)
        learning_times = self._benchmark_learning(agent, environment)
        
        # Test memory usage
        memory_usage = self._benchmark_memory_usage(agent, environment)
        
        # Compile metrics
        metrics = {
            'action_generation_time_ms': np.mean(action_times) * 1000,
            'action_generation_std_ms': np.std(action_times) * 1000,
            'actions_per_second': 1.0 / np.mean(action_times) if action_times else 0,
            'memory_usage_mb': memory_usage,
        }
        
        if learning_times:
            metrics.update({
                'learning_time_ms': np.mean(learning_times) * 1000,
                'learning_std_ms': np.std(learning_times) * 1000,
                'learning_steps_per_second': 1.0 / np.mean(learning_times)
            })
        
        return metrics
    
    def _benchmark_action_generation(self, agent, obs) -> List[float]:
        """Benchmark action generation speed."""
        times = []
        
        # Warmup
        for _ in range(10):
            agent.act(obs)
        
        # Benchmark
        for _ in range(self.num_samples):
            start_time = time.perf_counter()
            action = agent.act(obs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return times
    
    def _benchmark_learning(self, agent, environment) -> List[float]:
        """Benchmark learning speed."""
        if not hasattr(agent, 'learn'):
            return []
        
        times = []
        obs = environment.reset()
        
        # Warmup
        for _ in range(10):
            action = agent.act(obs)
            next_obs, reward, done, info = environment.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs if not done else environment.reset()
        
        # Benchmark
        for _ in range(min(self.num_samples, 100)):  # Fewer samples for learning
            action = agent.act(obs)
            next_obs, reward, done, info = environment.step(action)
            
            start_time = time.perf_counter()
            agent.learn(obs, action, reward, next_obs, done)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            obs = next_obs if not done else environment.reset()
        
        return times
    
    def _benchmark_memory_usage(self, agent, environment) -> float:
        """Benchmark memory usage."""
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Run some operations
        obs = environment.reset()
        for _ in range(100):
            action = agent.act(obs)
            obs, reward, done, info = environment.step(action)
            
            if hasattr(agent, 'learn'):
                agent.learn(obs, action, reward, obs, done)
            
            if done:
                obs = environment.reset()
        
        # Measure memory after operations
        current_memory = process.memory_info().rss / 1024 / 1024
        
        return current_memory - baseline_memory
    
    def cleanup(self):
        """Cleanup efficiency benchmark."""
        gc.collect()


class StabilityBenchmark(BaseBenchmark):
    """Benchmark for training stability and convergence."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_runs = config.get('num_runs', 5)
        self.episodes_per_run = config.get('episodes_per_run', 100)
        self.convergence_window = config.get('convergence_window', 20)
        self.convergence_threshold = config.get('convergence_threshold', 0.05)
        
    def setup(self, agent, environment):
        """Setup stability benchmark."""
        self.run_results = []
        
    def run_benchmark(self, agent, environment) -> Dict[str, float]:
        """Run stability benchmark."""
        logger.info(f"Running stability benchmark with {self.num_runs} runs")
        
        for run in range(self.num_runs):
            logger.debug(f"Stability run {run + 1}/{self.num_runs}")
            
            # Reset agent if possible
            if hasattr(agent, 'reset'):
                agent.reset()
            
            run_rewards = []
            obs = environment.reset()
            
            for episode in range(self.episodes_per_run):
                episode_reward = 0
                
                for step in range(1000):  # Max steps per episode
                    action = agent.act(obs)
                    next_obs, reward, done, info = environment.step(action)
                    
                    if hasattr(agent, 'learn'):
                        agent.learn(obs, action, reward, next_obs, done)
                    
                    episode_reward += reward
                    obs = next_obs
                    
                    if done:
                        obs = environment.reset()
                        break
                
                run_rewards.append(episode_reward)
            
            self.run_results.append(run_rewards)
        
        # Analyze stability
        metrics = self._analyze_stability()
        
        return metrics
    
    def _analyze_stability(self) -> Dict[str, float]:
        """Analyze training stability."""
        # Convert to numpy array
        results_array = np.array(self.run_results)  # [num_runs, episodes_per_run]
        
        # Final performance statistics
        final_rewards = results_array[:, -self.convergence_window:]
        final_means = np.mean(final_rewards, axis=1)
        
        # Convergence analysis
        convergence_episodes = []
        for run_rewards in self.run_results:
            conv_episode = self._find_convergence_episode(run_rewards)
            convergence_episodes.append(conv_episode)
        
        # Learning curve variance
        mean_curve = np.mean(results_array, axis=0)
        std_curve = np.std(results_array, axis=0)
        cv_curve = std_curve / (np.abs(mean_curve) + 1e-8)  # Coefficient of variation
        
        metrics = {
            'final_performance_mean': np.mean(final_means),
            'final_performance_std': np.std(final_means),
            'final_performance_cv': np.std(final_means) / (np.abs(np.mean(final_means)) + 1e-8),
            'convergence_episode_mean': np.mean([ep for ep in convergence_episodes if ep is not None]),
            'convergence_episode_std': np.std([ep for ep in convergence_episodes if ep is not None]),
            'convergence_success_rate': np.mean([ep is not None for ep in convergence_episodes]),
            'learning_curve_cv_mean': np.mean(cv_curve),
            'learning_curve_cv_final': np.mean(cv_curve[-self.convergence_window:]),
            'stability_score': self._calculate_stability_score(results_array)
        }
        
        return metrics
    
    def _find_convergence_episode(self, rewards: List[float]) -> Optional[int]:
        """Find episode where training converged."""
        if len(rewards) < self.convergence_window * 2:
            return None
        
        for i in range(self.convergence_window, len(rewards) - self.convergence_window):
            window_before = rewards[i-self.convergence_window:i]
            window_after = rewards[i:i+self.convergence_window]
            
            mean_before = np.mean(window_before)
            mean_after = np.mean(window_after)
            
            if abs(mean_after - mean_before) / (abs(mean_before) + 1e-8) < self.convergence_threshold:
                return i
        
        return None
    
    def _calculate_stability_score(self, results_array: np.ndarray) -> float:
        """Calculate overall stability score (0-1, higher is better)."""
        # Combine multiple stability indicators
        final_rewards = results_array[:, -self.convergence_window:]
        final_cv = np.std(np.mean(final_rewards, axis=1)) / (np.abs(np.mean(final_rewards)) + 1e-8)
        
        # Learning curve smoothness
        mean_curve = np.mean(results_array, axis=0)
        curve_smoothness = 1.0 / (1.0 + np.std(np.diff(mean_curve)))
        
        # Convergence consistency
        convergence_episodes = [self._find_convergence_episode(run) for run in results_array]
        convergence_rate = np.mean([ep is not None for ep in convergence_episodes])
        
        # Combined score
        stability_score = (
            (1.0 / (1.0 + final_cv)) * 0.4 +
            curve_smoothness * 0.3 +
            convergence_rate * 0.3
        )
        
        return stability_score
    
    def cleanup(self):
        """Cleanup stability benchmark."""
        pass


class BenchmarkRunner:
    """Main benchmark runner for comprehensive evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize benchmark runner.
        
        Args:
            config: Runner configuration
        """
        self.config = config
        self.benchmarks: Dict[str, BaseBenchmark] = {}
        self.results: List[BenchmarkResult] = []
        
        # Configuration
        self.output_dir = Path(config.get('output_dir', './benchmark_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.parallel_execution = config.get('parallel_execution', False)
        self.max_workers = config.get('max_workers', 2)
        
        logger.info("Initialized BenchmarkRunner")
    
    def register_benchmark(self, name: str, benchmark: BaseBenchmark):
        """Register a benchmark."""
        self.benchmarks[name] = benchmark
        logger.info(f"Registered benchmark: {name}")
    
    def run_benchmark(self, benchmark_name: str, agent, environment) -> BenchmarkResult:
        """Run a specific benchmark."""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Benchmark '{benchmark_name}' not registered")
        
        benchmark = self.benchmarks[benchmark_name]
        logger.info(f"Running benchmark: {benchmark_name}")
        
        result = benchmark.execute(agent, environment)
        self.results.append(result)
        
        return result
    
    def run_all_benchmarks(self, agent, environment) -> List[BenchmarkResult]:
        """Run all registered benchmarks."""
        logger.info(f"Running {len(self.benchmarks)} benchmarks")
        
        results = []
        
        if self.parallel_execution:
            results = self._run_benchmarks_parallel(agent, environment)
        else:
            for name, benchmark in self.benchmarks.items():
                result = self.run_benchmark(name, agent, environment)
                results.append(result)
        
        # Generate comprehensive report
        self._generate_report(results)
        
        return results
    
    def _run_benchmarks_parallel(self, agent, environment) -> List[BenchmarkResult]:
        """Run benchmarks in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit benchmarks
            future_to_name = {
                executor.submit(self.run_benchmark, name, agent, environment): name
                for name in self.benchmarks.keys()
            }
            
            # Collect results
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Benchmark {name} failed: {e}")
        
        return results
    
    def compare_agents(self, agents: List[Tuple[str, Any]], environment) -> pd.DataFrame:
        """Compare multiple agents on all benchmarks."""
        logger.info(f"Comparing {len(agents)} agents")
        
        comparison_results = []
        
        for agent_name, agent in agents:
            logger.info(f"Benchmarking agent: {agent_name}")
            
            agent_results = self.run_all_benchmarks(agent, environment)
            
            for result in agent_results:
                row = {
                    'agent': agent_name,
                    'benchmark': result.benchmark_name,
                    'execution_time': result.execution_time,
                    **result.metrics
                }
                comparison_results.append(row)
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_results)
        
        # Save comparison results
        comparison_file = self.output_dir / f'agent_comparison_{int(time.time())}.csv'
        df.to_csv(comparison_file, index=False)
        
        logger.info(f"Agent comparison saved to {comparison_file}")
        
        return df
    
    def _generate_report(self, results: List[BenchmarkResult]):
        """Generate comprehensive benchmark report."""
        if not results:
            return
        
        # Create report data
        report_data = {
            'summary': {
                'total_benchmarks': len(results),
                'total_execution_time': sum(r.execution_time for r in results),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': []
        }
        
        for result in results:
            report_data['results'].append({
                'benchmark_name': result.benchmark_name,
                'agent_name': result.agent_name,
                'environment_name': result.environment_name,
                'execution_time': result.execution_time,
                'metrics': result.metrics,
                'timestamp': result.timestamp
            })
        
        # Save detailed report
        report_file = self.output_dir / f'benchmark_report_{int(time.time())}.json'
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Generate summary statistics
        self._generate_summary_stats(results)
        
        logger.info(f"Benchmark report saved to {report_file}")
    
    def _generate_summary_stats(self, results: List[BenchmarkResult]):
        """Generate summary statistics."""
        # Collect all metrics
        all_metrics = defaultdict(list)
        
        for result in results:
            for metric_name, value in result.metrics.items():
                all_metrics[metric_name].append(value)
        
        # Calculate summary statistics
        summary_stats = {}
        for metric_name, values in all_metrics.items():
            summary_stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        # Save summary
        summary_file = self.output_dir / f'benchmark_summary_{int(time.time())}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        logger.info(f"Benchmark summary saved to {summary_file}")
    
    def visualize_results(self, results: List[BenchmarkResult], save_plots: bool = True):
        """Create visualizations of benchmark results."""
        if not results:
            return
        
        # Create performance comparison plot
        self._plot_performance_comparison(results, save_plots)
        
        # Create efficiency comparison plot
        self._plot_efficiency_comparison(results, save_plots)
        
        # Create stability analysis plot
        self._plot_stability_analysis(results, save_plots)
    
    def _plot_performance_comparison(self, results: List[BenchmarkResult], save: bool):
        """Plot performance comparison."""
        perf_results = [r for r in results if 'mean_reward' in r.metrics]
        
        if not perf_results:
            return
        
        agents = [r.agent_name for r in perf_results]
        rewards = [r.metrics['mean_reward'] for r in perf_results]
        errors = [r.metrics.get('std_reward', 0) for r in perf_results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(agents, rewards, yerr=errors, capsize=5)
        plt.title('Agent Performance Comparison')
        plt.xlabel('Agent')
        plt.ylabel('Mean Reward')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_efficiency_comparison(self, results: List[BenchmarkResult], save: bool):
        """Plot efficiency comparison."""
        eff_results = [r for r in results if 'actions_per_second' in r.metrics]
        
        if not eff_results:
            return
        
        agents = [r.agent_name for r in eff_results]
        aps = [r.metrics['actions_per_second'] for r in eff_results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(agents, aps)
        plt.title('Agent Efficiency Comparison')
        plt.xlabel('Agent')
        plt.ylabel('Actions per Second')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_stability_analysis(self, results: List[BenchmarkResult], save: bool):
        """Plot stability analysis."""
        stab_results = [r for r in results if 'stability_score' in r.metrics]
        
        if not stab_results:
            return
        
        agents = [r.agent_name for r in stab_results]
        scores = [r.metrics['stability_score'] for r in stab_results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(agents, scores)
        plt.title('Agent Stability Analysis')
        plt.xlabel('Agent')
        plt.ylabel('Stability Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'stability_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.show()


def create_standard_benchmark_suite() -> Dict[str, BaseBenchmark]:
    """Create standard benchmark suite."""
    return {
        'performance': PerformanceBenchmark({'num_episodes': 50}),
        'efficiency': EfficiencyBenchmark({'num_samples': 1000}),
        'stability': StabilityBenchmark({'num_runs': 3, 'episodes_per_run': 50})
    }

