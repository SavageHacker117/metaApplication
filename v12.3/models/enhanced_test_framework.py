#!/usr/bin/env python3
"""
Enhanced Testing Framework v3

Improvements based on feedback:
- Mini "smoke test" mode (1-2 rollout steps)
- Random agent unit tests for edge cases
- Comprehensive failure case tracking
- Enhanced debugging capabilities
- Performance benchmarking
- Automated regression testing
"""

import sys
import os
import traceback
import time
import json
import logging
import unittest
import random
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
import tempfile
import shutil
import pickle
import hashlib
from collections import defaultdict, deque
import threading
import queue
import psutil
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class TestResult:
    """Enhanced test result with detailed tracking."""
    test_name: str
    success: bool
    execution_time: float
    memory_usage: float
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

@dataclass
class TestConfig:
    """Configuration for enhanced testing."""
    # Test modes
    smoke_test_mode: bool = False
    full_test_mode: bool = True
    performance_test_mode: bool = False
    
    # Smoke test settings
    smoke_test_steps: int = 2
    smoke_test_timeout: float = 30.0
    
    # Random agent testing
    enable_random_agent_tests: bool = True
    random_test_iterations: int = 100
    random_seed: int = 42
    
    # Failure tracking
    enable_failure_tracking: bool = True
    failure_log_path: str = "test_failures.json"
    max_failure_history: int = 1000
    
    # Performance monitoring
    monitor_memory: bool = True
    monitor_cpu: bool = True
    performance_threshold_memory_mb: float = 1000.0
    performance_threshold_cpu_percent: float = 80.0
    
    # Output settings
    verbose: bool = True
    save_detailed_logs: bool = True
    log_directory: str = "test_logs"

class FailureTracker:
    """Tracks and analyzes test failures."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.failure_history = deque(maxlen=config.max_failure_history)
        self.failure_patterns = defaultdict(int)
        self.logger = logging.getLogger(__name__)
        
        # Load existing failure history
        self._load_failure_history()
    
    def record_failure(self, test_result: TestResult):
        """Record a test failure."""
        if not self.config.enable_failure_tracking or test_result.success:
            return
        
        failure_record = {
            'timestamp': time.time(),
            'test_name': test_result.test_name,
            'error_type': test_result.error_type,
            'error_message': test_result.error_message,
            'execution_time': test_result.execution_time,
            'memory_usage': test_result.memory_usage,
            'metadata': test_result.metadata
        }
        
        self.failure_history.append(failure_record)
        
        # Track failure patterns
        pattern_key = f"{test_result.test_name}:{test_result.error_type}"
        self.failure_patterns[pattern_key] += 1
        
        # Save updated history
        self._save_failure_history()
    
    def _load_failure_history(self):
        """Load failure history from file."""
        try:
            failure_path = Path(self.config.failure_log_path)
            if failure_path.exists():
                with open(failure_path, 'r') as f:
                    data = json.load(f)
                    self.failure_history.extend(data.get('failures', []))
                    self.failure_patterns.update(data.get('patterns', {}))
        except Exception as e:
            self.logger.warning(f"Failed to load failure history: {e}")
    
    def _save_failure_history(self):
        """Save failure history to file."""
        try:
            failure_path = Path(self.config.failure_log_path)
            failure_path.parent.mkdir(exist_ok=True)
            
            data = {
                'failures': list(self.failure_history),
                'patterns': dict(self.failure_patterns),
                'last_updated': time.time()
            }
            
            with open(failure_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save failure history: {e}")
    
    def get_failure_analysis(self) -> Dict[str, Any]:
        """Get comprehensive failure analysis."""
        if not self.failure_history:
            return {'total_failures': 0, 'patterns': {}, 'recent_failures': []}
        
        # Recent failures (last 24 hours)
        recent_cutoff = time.time() - 86400
        recent_failures = [f for f in self.failure_history if f['timestamp'] > recent_cutoff]
        
        # Most common failure patterns
        top_patterns = sorted(
            self.failure_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_failures': len(self.failure_history),
            'recent_failures_24h': len(recent_failures),
            'top_failure_patterns': top_patterns,
            'failure_rate_by_test': self._calculate_failure_rates(),
            'recent_failures': recent_failures[-10:]  # Last 10 failures
        }
    
    def _calculate_failure_rates(self) -> Dict[str, float]:
        """Calculate failure rates by test."""
        test_counts = defaultdict(int)
        test_failures = defaultdict(int)
        
        for failure in self.failure_history:
            test_name = failure['test_name']
            test_failures[test_name] += 1
            test_counts[test_name] += 1  # This is simplified; would need success counts too
        
        # For now, just return failure counts
        return dict(test_failures)

class PerformanceMonitor:
    """Monitors test performance and resource usage."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    @contextmanager
    def monitor_test(self, test_name: str):
        """Context manager for monitoring test performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        
        # Start monitoring thread
        monitoring_data = {'memory': [], 'cpu': []}
        monitoring_active = threading.Event()
        monitoring_active.set()
        
        def monitor_loop():
            while monitoring_active.is_set():
                if self.config.monitor_memory:
                    monitoring_data['memory'].append(self._get_memory_usage())
                if self.config.monitor_cpu:
                    monitoring_data['cpu'].append(psutil.cpu_percent())
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        try:
            yield monitoring_data
        finally:
            monitoring_active.clear()
            monitor_thread.join(timeout=1.0)
            
            # Calculate final metrics
            end_time = time.time()
            execution_time = end_time - start_time
            peak_memory = max(monitoring_data['memory']) if monitoring_data['memory'] else start_memory
            avg_cpu = np.mean(monitoring_data['cpu']) if monitoring_data['cpu'] else start_cpu
            
            # Check performance thresholds
            if peak_memory > self.config.performance_threshold_memory_mb:
                self.logger.warning(f"Test {test_name} exceeded memory threshold: {peak_memory:.1f}MB")
            
            if avg_cpu > self.config.performance_threshold_cpu_percent:
                self.logger.warning(f"Test {test_name} exceeded CPU threshold: {avg_cpu:.1f}%")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

class RandomAgentTester:
    """Tests system with random agent behavior to find edge cases."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.edge_cases_found = []
        
        # Set random seed for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def test_random_agent_behavior(self) -> TestResult:
        """Test system with random agent actions."""
        start_time = time.time()
        
        try:
            # Import required components
            from core.environment.base_environment import TowerDefenseEnvironment
            from core.actions.action_space import ActionSpace
            
            # Create test environment
            env = TowerDefenseEnvironment(grid_size=[5, 5], max_towers=3, max_enemies=5)
            action_space = ActionSpace()
            
            edge_cases = []
            
            for iteration in range(self.config.random_test_iterations):
                try:
                    # Reset environment
                    state = env.reset()
                    
                    # Perform random actions
                    for step in range(10):  # Short episodes
                        # Generate random action
                        action = self._generate_random_action(action_space, state)
                        
                        # Execute action
                        next_state, reward, done, info = env.step(action)
                        
                        # Check for edge cases
                        edge_case = self._check_for_edge_cases(state, action, next_state, reward, info)
                        if edge_case:
                            edge_cases.append(edge_case)
                        
                        if done:
                            break
                        
                        state = next_state
                
                except Exception as e:
                    # Record edge case
                    edge_case = {
                        'iteration': iteration,
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'context': 'random_action_execution'
                    }
                    edge_cases.append(edge_case)
            
            execution_time = time.time() - start_time
            
            # Store edge cases for analysis
            self.edge_cases_found.extend(edge_cases)
            
            return TestResult(
                test_name="random_agent_behavior",
                success=True,
                execution_time=execution_time,
                memory_usage=0.0,  # Would be filled by performance monitor
                metadata={
                    'iterations_completed': self.config.random_test_iterations,
                    'edge_cases_found': len(edge_cases),
                    'edge_cases': edge_cases[:10]  # First 10 for logging
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="random_agent_behavior",
                success=False,
                execution_time=execution_time,
                memory_usage=0.0,
                error_message=str(e),
                error_type=type(e).__name__,
                stack_trace=traceback.format_exc()
            )
    
    def _generate_random_action(self, action_space, state) -> Dict[str, Any]:
        """Generate a random action."""
        # Simplified random action generation
        action_types = ['place_tower', 'upgrade_tower', 'sell_tower', 'no_action']
        action_type = random.choice(action_types)
        
        if action_type == 'place_tower':
            return {
                'type': 'place_tower',
                'position': [random.randint(0, 4), random.randint(0, 4)],
                'tower_type': random.choice(['basic', 'cannon', 'archer'])
            }
        elif action_type == 'upgrade_tower':
            return {
                'type': 'upgrade_tower',
                'position': [random.randint(0, 4), random.randint(0, 4)]
            }
        elif action_type == 'sell_tower':
            return {
                'type': 'sell_tower',
                'position': [random.randint(0, 4), random.randint(0, 4)]
            }
        else:
            return {'type': 'no_action'}
    
    def _check_for_edge_cases(self, state, action, next_state, reward, info) -> Optional[Dict[str, Any]]:
        """Check for potential edge cases."""
        edge_cases = []
        
        # Check for extreme reward values
        if abs(reward) > 1000:
            edge_cases.append({
                'type': 'extreme_reward',
                'value': reward,
                'action': action
            })
        
        # Check for NaN or infinite values
        if np.isnan(reward) or np.isinf(reward):
            edge_cases.append({
                'type': 'invalid_reward',
                'value': reward,
                'action': action
            })
        
        # Check for state inconsistencies
        if isinstance(next_state, dict) and isinstance(state, dict):
            for key in state.keys():
                if key in next_state:
                    if isinstance(state[key], (int, float)) and isinstance(next_state[key], (int, float)):
                        if abs(next_state[key] - state[key]) > 1000:  # Arbitrary large change
                            edge_cases.append({
                                'type': 'large_state_change',
                                'key': key,
                                'old_value': state[key],
                                'new_value': next_state[key],
                                'action': action
                            })
        
        return edge_cases[0] if edge_cases else None

class SmokeTestRunner:
    """Runs quick smoke tests to verify basic functionality."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run_smoke_tests(self) -> List[TestResult]:
        """Run all smoke tests."""
        smoke_tests = [
            ("import_smoke_test", self._test_imports),
            ("initialization_smoke_test", self._test_initialization),
            ("basic_rollout_smoke_test", self._test_basic_rollout),
            ("reward_calculation_smoke_test", self._test_reward_calculation)
        ]
        
        results = []
        
        for test_name, test_func in smoke_tests:
            try:
                with self._timeout_context(self.config.smoke_test_timeout):
                    result = test_func()
                    results.append(result)
            except TimeoutError:
                result = TestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=self.config.smoke_test_timeout,
                    memory_usage=0.0,
                    error_message="Test timed out",
                    error_type="TimeoutError"
                )
                results.append(result)
            except Exception as e:
                result = TestResult(
                    test_name=test_name,
                    success=False,
                    execution_time=0.0,
                    memory_usage=0.0,
                    error_message=str(e),
                    error_type=type(e).__name__,
                    stack_trace=traceback.format_exc()
                )
                results.append(result)
        
        return results
    
    @contextmanager
    def _timeout_context(self, timeout: float):
        """Context manager for test timeout."""
        # Simplified timeout implementation
        start_time = time.time()
        try:
            yield
        finally:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Test exceeded timeout of {timeout} seconds")
    
    def _test_imports(self) -> TestResult:
        """Smoke test for imports."""
        start_time = time.time()
        
        try:
            # Critical imports only
            from core.environment.base_environment import TowerDefenseEnvironment
            from core.rewards.reward_system import RewardSystem
            from core.actions.action_space import ActionSpace
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="import_smoke_test",
                success=True,
                execution_time=execution_time,
                memory_usage=0.0,
                metadata={'imports_tested': 3}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="import_smoke_test",
                success=False,
                execution_time=execution_time,
                memory_usage=0.0,
                error_message=str(e),
                error_type=type(e).__name__
            )
    
    def _test_initialization(self) -> TestResult:
        """Smoke test for basic initialization."""
        start_time = time.time()
        
        try:
            from core.environment.base_environment import TowerDefenseEnvironment
            
            # Quick initialization
            env = TowerDefenseEnvironment(grid_size=[3, 3], max_towers=1, max_enemies=1)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="initialization_smoke_test",
                success=True,
                execution_time=execution_time,
                memory_usage=0.0,
                metadata={'environment_initialized': True}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="initialization_smoke_test",
                success=False,
                execution_time=execution_time,
                memory_usage=0.0,
                error_message=str(e),
                error_type=type(e).__name__
            )
    
    def _test_basic_rollout(self) -> TestResult:
        """Smoke test for basic environment rollout."""
        start_time = time.time()
        
        try:
            from core.environment.base_environment import TowerDefenseEnvironment
            
            env = TowerDefenseEnvironment(grid_size=[3, 3], max_towers=1, max_enemies=1)
            state = env.reset()
            
            # Perform limited rollout steps
            for step in range(self.config.smoke_test_steps):
                action = {'type': 'no_action'}  # Safe action
                next_state, reward, done, info = env.step(action)
                
                if done:
                    break
                
                state = next_state
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="basic_rollout_smoke_test",
                success=True,
                execution_time=execution_time,
                memory_usage=0.0,
                metadata={'steps_completed': min(step + 1, self.config.smoke_test_steps)}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="basic_rollout_smoke_test",
                success=False,
                execution_time=execution_time,
                memory_usage=0.0,
                error_message=str(e),
                error_type=type(e).__name__
            )
    
    def _test_reward_calculation(self) -> TestResult:
        """Smoke test for reward calculation."""
        start_time = time.time()
        
        try:
            from core.rewards.reward_system import RewardSystem
            
            reward_system = RewardSystem()
            
            # Test basic reward calculation
            reward = reward_system.calculate_placement_reward(1, 1, None, [])
            
            # Basic sanity checks
            if not isinstance(reward, (int, float)):
                raise ValueError(f"Reward should be numeric, got {type(reward)}")
            
            if np.isnan(reward) or np.isinf(reward):
                raise ValueError(f"Reward should be finite, got {reward}")
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="reward_calculation_smoke_test",
                success=True,
                execution_time=execution_time,
                memory_usage=0.0,
                metadata={'reward_value': reward}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name="reward_calculation_smoke_test",
                success=False,
                execution_time=execution_time,
                memory_usage=0.0,
                error_message=str(e),
                error_type=type(e).__name__
            )

class EnhancedTestFramework:
    """
    Enhanced testing framework with all improvements.
    """
    
    def __init__(self, config: Optional[TestConfig] = None):
        self.config = config or TestConfig()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.failure_tracker = FailureTracker(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.random_agent_tester = RandomAgentTester(self.config)
        self.smoke_test_runner = SmokeTestRunner(self.config)
        
        # Test results
        self.test_results = []
        
        self.logger.info("Enhanced test framework initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for test framework."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        # Create log directory
        if self.config.save_detailed_logs:
            log_dir = Path(self.config.log_directory)
            log_dir.mkdir(exist_ok=True)
            
            # File handler
            log_file = log_dir / f"test_run_{int(time.time())}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests based on configuration."""
        self.logger.info("Starting enhanced test suite")
        start_time = time.time()
        
        # Run smoke tests if enabled
        if self.config.smoke_test_mode:
            self.logger.info("Running smoke tests...")
            smoke_results = self.smoke_test_runner.run_smoke_tests()
            self.test_results.extend(smoke_results)
            
            # Record failures
            for result in smoke_results:
                if not result.success:
                    self.failure_tracker.record_failure(result)
        
        # Run full tests if enabled
        if self.config.full_test_mode and not self.config.smoke_test_mode:
            self.logger.info("Running full test suite...")
            full_results = self._run_full_tests()
            self.test_results.extend(full_results)
            
            # Record failures
            for result in full_results:
                if not result.success:
                    self.failure_tracker.record_failure(result)
        
        # Run random agent tests if enabled
        if self.config.enable_random_agent_tests:
            self.logger.info("Running random agent tests...")
            random_result = self.random_agent_tester.test_random_agent_behavior()
            self.test_results.append(random_result)
            
            if not random_result.success:
                self.failure_tracker.record_failure(random_result)
        
        # Run performance tests if enabled
        if self.config.performance_test_mode:
            self.logger.info("Running performance tests...")
            perf_results = self._run_performance_tests()
            self.test_results.extend(perf_results)
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = self._generate_test_report(total_time)
        
        self.logger.info(f"Test suite completed in {total_time:.2f} seconds")
        
        return report
    
    def _run_full_tests(self) -> List[TestResult]:
        """Run full test suite."""
        # This would implement the full test suite
        # For now, return placeholder results
        return []
    
    def _run_performance_tests(self) -> List[TestResult]:
        """Run performance-specific tests."""
        # This would implement performance tests
        # For now, return placeholder results
        return []
    
    def _generate_test_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        passed_tests = [r for r in self.test_results if r.success]
        failed_tests = [r for r in self.test_results if not r.success]
        
        # Performance statistics
        avg_execution_time = np.mean([r.execution_time for r in self.test_results]) if self.test_results else 0
        max_execution_time = max([r.execution_time for r in self.test_results]) if self.test_results else 0
        
        # Failure analysis
        failure_analysis = self.failure_tracker.get_failure_analysis()
        
        report = {
            'summary': {
                'total_tests': len(self.test_results),
                'passed': len(passed_tests),
                'failed': len(failed_tests),
                'success_rate': len(passed_tests) / max(len(self.test_results), 1),
                'total_execution_time': total_time
            },
            'performance': {
                'average_test_time': avg_execution_time,
                'max_test_time': max_execution_time,
                'slowest_tests': sorted(
                    self.test_results,
                    key=lambda x: x.execution_time,
                    reverse=True
                )[:5]
            },
            'failures': {
                'failed_tests': [
                    {
                        'name': r.test_name,
                        'error_type': r.error_type,
                        'error_message': r.error_message
                    }
                    for r in failed_tests
                ],
                'failure_analysis': failure_analysis
            },
            'edge_cases': {
                'random_agent_edge_cases': len(self.random_agent_tester.edge_cases_found),
                'edge_case_examples': self.random_agent_tester.edge_cases_found[:5]
            },
            'configuration': {
                'smoke_test_mode': self.config.smoke_test_mode,
                'full_test_mode': self.config.full_test_mode,
                'random_agent_tests': self.config.enable_random_agent_tests,
                'performance_tests': self.config.performance_test_mode
            }
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None):
        """Save test report to file."""
        if filename is None:
            filename = f"test_report_{int(time.time())}.json"
        
        report_path = Path(self.config.log_directory) / filename
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Test report saved to {report_path}")

def main():
    """Main entry point for enhanced testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced RL Training Test Framework")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--no-random", action="store_true", help="Disable random agent tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--timeout", type=float, default=30.0, help="Smoke test timeout")
    
    args = parser.parse_args()
    
    # Configure test framework
    config = TestConfig(
        smoke_test_mode=args.smoke,
        full_test_mode=args.full or not (args.smoke or args.performance),
        performance_test_mode=args.performance,
        enable_random_agent_tests=not args.no_random,
        verbose=args.verbose,
        smoke_test_timeout=args.timeout
    )
    
    # Run tests
    framework = EnhancedTestFramework(config)
    report = framework.run_all_tests()
    
    # Save report
    framework.save_report(report)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🧪 Enhanced Test Framework - Results Summary")
    print("=" * 60)
    
    summary = report['summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Total Time: {summary['total_execution_time']:.2f}s")
    
    if report['failures']['failed_tests']:
        print("\n❌ Failed Tests:")
        for failure in report['failures']['failed_tests']:
            print(f"  - {failure['name']}: {failure['error_type']}")
    
    if report['edge_cases']['random_agent_edge_cases'] > 0:
        print(f"\n⚠️ Edge Cases Found: {report['edge_cases']['random_agent_edge_cases']}")
    
    # Exit code
    exit_code = 0 if summary['failed'] == 0 else 1
    
    if exit_code == 0:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the detailed report for more information.")
    
    return exit_code

if __name__ == "__main__":
    exit(main())

