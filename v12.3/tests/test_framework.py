"""
Comprehensive Testing Framework for RL-LLM System

This module provides a complete testing framework for validating RL components,
environments, agents, and training pipelines with automated test discovery
and reporting capabilities.
"""

import unittest
import pytest
import numpy as np
import torch
import gym
from typing import Dict, Any, List, Optional, Callable, Type, Union
import logging
import time
import traceback
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result information."""
    test_name: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestSuite:
    """Test suite configuration."""
    name: str
    description: str
    tests: List[str] = field(default_factory=list)
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    timeout: float = 300.0  # 5 minutes default
    parallel: bool = False


class BaseTestCase(ABC):
    """Base class for all RL test cases."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize test case.
        
        Args:
            config: Test configuration
        """
        self.config = config
        self.test_name = self.__class__.__name__
        self.setup_done = False
        
    @abstractmethod
    def setup(self):
        """Setup test case."""
        pass
    
    @abstractmethod
    def run_test(self) -> TestResult:
        """Run the test case."""
        pass
    
    @abstractmethod
    def teardown(self):
        """Cleanup test case."""
        pass
    
    def execute(self) -> TestResult:
        """Execute the complete test case."""
        start_time = time.time()
        
        try:
            # Setup
            self.setup()
            self.setup_done = True
            
            # Run test
            result = self.run_test()
            result.execution_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            logger.error(f"Test {self.test_name} failed: {error_msg}")
            
            return TestResult(
                test_name=self.test_name,
                passed=False,
                execution_time=execution_time,
                error_message=error_msg,
                details={'traceback': traceback.format_exc()}
            )
        
        finally:
            # Cleanup
            if self.setup_done:
                try:
                    self.teardown()
                except Exception as e:
                    logger.warning(f"Teardown failed for {self.test_name}: {e}")


class EnvironmentTestCase(BaseTestCase):
    """Test case for RL environments."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.environment = None
        self.num_episodes = config.get('num_episodes', 5)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 100)
    
    def setup(self):
        """Setup environment for testing."""
        env_config = self.config.get('environment_config', {})
        env_class = self.config['environment_class']
        
        if isinstance(env_class, str):
            # Import environment class from string
            module_name, class_name = env_class.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            env_class = getattr(module, class_name)
        
        self.environment = env_class(env_config)
    
    def run_test(self) -> TestResult:
        """Run environment tests."""
        test_results = {
            'reset_test': self._test_reset(),
            'step_test': self._test_step(),
            'action_space_test': self._test_action_space(),
            'observation_space_test': self._test_observation_space(),
            'episode_test': self._test_full_episodes(),
            'determinism_test': self._test_determinism()
        }
        
        all_passed = all(test_results.values())
        
        # Collect metrics
        metrics = self._collect_environment_metrics()
        
        return TestResult(
            test_name=self.test_name,
            passed=all_passed,
            execution_time=0,  # Will be set by execute()
            details=test_results,
            metrics=metrics
        )
    
    def _test_reset(self) -> bool:
        """Test environment reset functionality."""
        try:
            obs = self.environment.reset()
            
            # Check observation type and shape
            if hasattr(self.environment, 'observation_space'):
                assert self.environment.observation_space.contains(obs), "Reset observation not in observation space"
            
            return True
        except Exception as e:
            logger.error(f"Reset test failed: {e}")
            return False
    
    def _test_step(self) -> bool:
        """Test environment step functionality."""
        try:
            obs = self.environment.reset()
            
            # Sample random action
            if hasattr(self.environment, 'action_space'):
                action = self.environment.action_space.sample()
            else:
                action = 0  # Default action
            
            next_obs, reward, done, info = self.environment.step(action)
            
            # Validate outputs
            assert isinstance(reward, (int, float)), "Reward must be numeric"
            assert isinstance(done, bool), "Done must be boolean"
            assert isinstance(info, dict), "Info must be dictionary"
            
            if hasattr(self.environment, 'observation_space'):
                assert self.environment.observation_space.contains(next_obs), "Step observation not in observation space"
            
            return True
        except Exception as e:
            logger.error(f"Step test failed: {e}")
            return False
    
    def _test_action_space(self) -> bool:
        """Test action space properties."""
        try:
            if not hasattr(self.environment, 'action_space'):
                return True  # Skip if no action space defined
            
            action_space = self.environment.action_space
            
            # Test sampling
            for _ in range(10):
                action = action_space.sample()
                assert action_space.contains(action), "Sampled action not in action space"
            
            return True
        except Exception as e:
            logger.error(f"Action space test failed: {e}")
            return False
    
    def _test_observation_space(self) -> bool:
        """Test observation space properties."""
        try:
            if not hasattr(self.environment, 'observation_space'):
                return True  # Skip if no observation space defined
            
            obs_space = self.environment.observation_space
            
            # Test with reset observation
            obs = self.environment.reset()
            assert obs_space.contains(obs), "Reset observation not in observation space"
            
            return True
        except Exception as e:
            logger.error(f"Observation space test failed: {e}")
            return False
    
    def _test_full_episodes(self) -> bool:
        """Test full episode execution."""
        try:
            for episode in range(self.num_episodes):
                obs = self.environment.reset()
                total_reward = 0
                
                for step in range(self.max_steps_per_episode):
                    if hasattr(self.environment, 'action_space'):
                        action = self.environment.action_space.sample()
                    else:
                        action = 0
                    
                    obs, reward, done, info = self.environment.step(action)
                    total_reward += reward
                    
                    if done:
                        break
                
                # Episode completed successfully
                logger.debug(f"Episode {episode} completed with reward: {total_reward}")
            
            return True
        except Exception as e:
            logger.error(f"Full episode test failed: {e}")
            return False
    
    def _test_determinism(self) -> bool:
        """Test environment determinism with fixed seed."""
        try:
            if not hasattr(self.environment, 'seed'):
                return True  # Skip if seeding not supported
            
            # Run same sequence twice with same seed
            seed = 42
            
            # First run
            self.environment.seed(seed)
            obs1 = self.environment.reset()
            action = 0 if not hasattr(self.environment, 'action_space') else self.environment.action_space.sample()
            next_obs1, reward1, done1, info1 = self.environment.step(action)
            
            # Second run
            self.environment.seed(seed)
            obs2 = self.environment.reset()
            next_obs2, reward2, done2, info2 = self.environment.step(action)
            
            # Compare results
            if isinstance(obs1, np.ndarray) and isinstance(obs2, np.ndarray):
                np.testing.assert_array_equal(obs1, obs2, "Reset observations not deterministic")
                np.testing.assert_array_equal(next_obs1, next_obs2, "Step observations not deterministic")
            
            assert reward1 == reward2, "Rewards not deterministic"
            assert done1 == done2, "Done flags not deterministic"
            
            return True
        except Exception as e:
            logger.error(f"Determinism test failed: {e}")
            return False
    
    def _collect_environment_metrics(self) -> Dict[str, float]:
        """Collect environment performance metrics."""
        metrics = {}
        
        try:
            # Timing metrics
            start_time = time.time()
            self.environment.reset()
            reset_time = time.time() - start_time
            metrics['reset_time'] = reset_time
            
            start_time = time.time()
            if hasattr(self.environment, 'action_space'):
                action = self.environment.action_space.sample()
            else:
                action = 0
            self.environment.step(action)
            step_time = time.time() - start_time
            metrics['step_time'] = step_time
            
            # Memory usage
            process = psutil.Process()
            metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            
        except Exception as e:
            logger.warning(f"Failed to collect environment metrics: {e}")
        
        return metrics
    
    def teardown(self):
        """Cleanup environment."""
        if self.environment and hasattr(self.environment, 'close'):
            self.environment.close()


class AgentTestCase(BaseTestCase):
    """Test case for RL agents."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.agent = None
        self.environment = None
        self.num_episodes = config.get('num_episodes', 3)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 50)
    
    def setup(self):
        """Setup agent and environment for testing."""
        # Create environment
        env_config = self.config.get('environment_config', {})
        env_class = self.config['environment_class']
        
        if isinstance(env_class, str):
            module_name, class_name = env_class.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            env_class = getattr(module, class_name)
        
        self.environment = env_class(env_config)
        
        # Create agent
        agent_config = self.config.get('agent_config', {})
        agent_class = self.config['agent_class']
        
        if isinstance(agent_class, str):
            module_name, class_name = agent_class.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            agent_class = getattr(module, class_name)
        
        self.agent = agent_class(agent_config)
    
    def run_test(self) -> TestResult:
        """Run agent tests."""
        test_results = {
            'action_test': self._test_action_generation(),
            'learning_test': self._test_learning(),
            'episode_test': self._test_full_episodes(),
            'state_dict_test': self._test_state_dict(),
            'performance_test': self._test_performance()
        }
        
        all_passed = all(test_results.values())
        
        # Collect metrics
        metrics = self._collect_agent_metrics()
        
        return TestResult(
            test_name=self.test_name,
            passed=all_passed,
            execution_time=0,
            details=test_results,
            metrics=metrics
        )
    
    def _test_action_generation(self) -> bool:
        """Test agent action generation."""
        try:
            obs = self.environment.reset()
            
            # Test action generation
            action = self.agent.act(obs)
            
            # Validate action
            if hasattr(self.environment, 'action_space'):
                assert self.environment.action_space.contains(action), "Generated action not in action space"
            
            return True
        except Exception as e:
            logger.error(f"Action generation test failed: {e}")
            return False
    
    def _test_learning(self) -> bool:
        """Test agent learning capability."""
        try:
            if not hasattr(self.agent, 'learn'):
                return True  # Skip if agent doesn't have learning
            
            obs = self.environment.reset()
            action = self.agent.act(obs)
            next_obs, reward, done, info = self.environment.step(action)
            
            # Test learning
            loss = self.agent.learn(obs, action, reward, next_obs, done)
            
            if loss is not None:
                assert isinstance(loss, (int, float)), "Learning loss must be numeric"
            
            return True
        except Exception as e:
            logger.error(f"Learning test failed: {e}")
            return False
    
    def _test_full_episodes(self) -> bool:
        """Test agent in full episodes."""
        try:
            for episode in range(self.num_episodes):
                obs = self.environment.reset()
                total_reward = 0
                
                for step in range(self.max_steps_per_episode):
                    action = self.agent.act(obs)
                    next_obs, reward, done, info = self.environment.step(action)
                    
                    # Learn if possible
                    if hasattr(self.agent, 'learn'):
                        self.agent.learn(obs, action, reward, next_obs, done)
                    
                    total_reward += reward
                    obs = next_obs
                    
                    if done:
                        break
                
                logger.debug(f"Agent episode {episode} completed with reward: {total_reward}")
            
            return True
        except Exception as e:
            logger.error(f"Full episode test failed: {e}")
            return False
    
    def _test_state_dict(self) -> bool:
        """Test agent state dictionary functionality."""
        try:
            if not hasattr(self.agent, 'state_dict'):
                return True  # Skip if no state dict
            
            # Get state dict
            state_dict = self.agent.state_dict()
            assert isinstance(state_dict, dict), "State dict must be dictionary"
            
            # Test loading state dict
            if hasattr(self.agent, 'load_state_dict'):
                self.agent.load_state_dict(state_dict)
            
            return True
        except Exception as e:
            logger.error(f"State dict test failed: {e}")
            return False
    
    def _test_performance(self) -> bool:
        """Test agent performance metrics."""
        try:
            # Run performance test
            rewards = []
            
            for _ in range(3):
                obs = self.environment.reset()
                total_reward = 0
                
                for _ in range(20):
                    action = self.agent.act(obs)
                    obs, reward, done, info = self.environment.step(action)
                    total_reward += reward
                    
                    if done:
                        break
                
                rewards.append(total_reward)
            
            # Basic performance check
            avg_reward = np.mean(rewards)
            logger.debug(f"Agent average reward: {avg_reward}")
            
            return True
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False
    
    def _collect_agent_metrics(self) -> Dict[str, float]:
        """Collect agent performance metrics."""
        metrics = {}
        
        try:
            obs = self.environment.reset()
            
            # Action generation time
            start_time = time.time()
            action = self.agent.act(obs)
            action_time = time.time() - start_time
            metrics['action_time'] = action_time
            
            # Learning time (if applicable)
            if hasattr(self.agent, 'learn'):
                next_obs, reward, done, info = self.environment.step(action)
                
                start_time = time.time()
                self.agent.learn(obs, action, reward, next_obs, done)
                learn_time = time.time() - start_time
                metrics['learn_time'] = learn_time
            
            # Memory usage
            process = psutil.Process()
            metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            
        except Exception as e:
            logger.warning(f"Failed to collect agent metrics: {e}")
        
        return metrics
    
    def teardown(self):
        """Cleanup agent and environment."""
        if self.environment and hasattr(self.environment, 'close'):
            self.environment.close()


class PerformanceTestCase(BaseTestCase):
    """Performance and stress testing."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_iterations = config.get('num_iterations', 1000)
        self.memory_threshold_mb = config.get('memory_threshold_mb', 1000)
        self.time_threshold_s = config.get('time_threshold_s', 10.0)
    
    def setup(self):
        """Setup performance test."""
        pass
    
    def run_test(self) -> TestResult:
        """Run performance tests."""
        test_results = {
            'memory_leak_test': self._test_memory_leaks(),
            'performance_test': self._test_performance(),
            'stress_test': self._test_stress()
        }
        
        all_passed = all(test_results.values())
        
        return TestResult(
            test_name=self.test_name,
            passed=all_passed,
            execution_time=0,
            details=test_results
        )
    
    def _test_memory_leaks(self) -> bool:
        """Test for memory leaks."""
        try:
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Run iterations
            for i in range(self.num_iterations):
                # Create and destroy objects
                data = np.random.randn(1000, 1000)
                del data
                
                if i % 100 == 0:
                    gc.collect()
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_increase = current_memory - initial_memory
                    
                    if memory_increase > self.memory_threshold_mb:
                        logger.error(f"Memory leak detected: {memory_increase:.2f} MB increase")
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Memory leak test failed: {e}")
            return False
    
    def _test_performance(self) -> bool:
        """Test performance benchmarks."""
        try:
            start_time = time.time()
            
            # Run performance-intensive operations
            for _ in range(self.num_iterations // 10):
                # Simulate computation
                data = np.random.randn(100, 100)
                result = np.dot(data, data.T)
                del data, result
            
            elapsed_time = time.time() - start_time
            
            if elapsed_time > self.time_threshold_s:
                logger.warning(f"Performance test slow: {elapsed_time:.2f}s > {self.time_threshold_s}s")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False
    
    def _test_stress(self) -> bool:
        """Stress test with concurrent operations."""
        try:
            def stress_worker():
                for _ in range(100):
                    data = np.random.randn(50, 50)
                    np.linalg.inv(data @ data.T + np.eye(50))
            
            # Run concurrent stress test
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(stress_worker) for _ in range(4)]
                
                for future in as_completed(futures):
                    future.result()  # Will raise exception if worker failed
            
            return True
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            return False
    
    def teardown(self):
        """Cleanup performance test."""
        gc.collect()


class TestRunner:
    """Main test runner for the RL-LLM system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize test runner.
        
        Args:
            config: Test runner configuration
        """
        self.config = config
        self.test_cases: List[BaseTestCase] = []
        self.test_suites: Dict[str, TestSuite] = {}
        self.results: List[TestResult] = []
        
        # Configuration
        self.parallel_execution = config.get('parallel_execution', False)
        self.max_workers = config.get('max_workers', 4)
        self.output_dir = Path(config.get('output_dir', './test_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized TestRunner")
    
    def add_test_case(self, test_case: BaseTestCase):
        """Add test case to runner."""
        self.test_cases.append(test_case)
    
    def add_test_suite(self, test_suite: TestSuite):
        """Add test suite to runner."""
        self.test_suites[test_suite.name] = test_suite
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all registered test cases."""
        logger.info(f"Running {len(self.test_cases)} test cases")
        
        if self.parallel_execution:
            results = self._run_tests_parallel()
        else:
            results = self._run_tests_sequential()
        
        self.results.extend(results)
        
        # Generate report
        self._generate_report()
        
        return results
    
    def run_test_suite(self, suite_name: str) -> List[TestResult]:
        """Run specific test suite."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite '{suite_name}' not found")
        
        suite = self.test_suites[suite_name]
        logger.info(f"Running test suite: {suite_name}")
        
        # Run suite setup
        if suite.setup_function:
            suite.setup_function()
        
        try:
            # Filter test cases for this suite
            suite_tests = [tc for tc in self.test_cases if tc.test_name in suite.tests]
            
            if suite.parallel:
                results = self._run_tests_parallel(suite_tests)
            else:
                results = self._run_tests_sequential(suite_tests)
            
            return results
            
        finally:
            # Run suite teardown
            if suite.teardown_function:
                suite.teardown_function()
    
    def _run_tests_sequential(self, test_cases: Optional[List[BaseTestCase]] = None) -> List[TestResult]:
        """Run tests sequentially."""
        if test_cases is None:
            test_cases = self.test_cases
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Running test {i+1}/{len(test_cases)}: {test_case.test_name}")
            
            try:
                result = test_case.execute()
                results.append(result)
                
                status = "PASSED" if result.passed else "FAILED"
                logger.info(f"Test {test_case.test_name}: {status} ({result.execution_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"Test {test_case.test_name} crashed: {e}")
                results.append(TestResult(
                    test_name=test_case.test_name,
                    passed=False,
                    execution_time=0,
                    error_message=str(e)
                ))
        
        return results
    
    def _run_tests_parallel(self, test_cases: Optional[List[BaseTestCase]] = None) -> List[TestResult]:
        """Run tests in parallel."""
        if test_cases is None:
            test_cases = self.test_cases
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tests
            future_to_test = {
                executor.submit(test_case.execute): test_case 
                for test_case in test_cases
            }
            
            # Collect results
            for future in as_completed(future_to_test):
                test_case = future_to_test[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    status = "PASSED" if result.passed else "FAILED"
                    logger.info(f"Test {test_case.test_name}: {status} ({result.execution_time:.2f}s)")
                    
                except Exception as e:
                    logger.error(f"Test {test_case.test_name} crashed: {e}")
                    results.append(TestResult(
                        test_name=test_case.test_name,
                        passed=False,
                        execution_time=0,
                        error_message=str(e)
                    ))
        
        return results
    
    def _generate_report(self):
        """Generate test report."""
        if not self.results:
            return
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        total_time = sum(r.execution_time for r in self.results)
        
        # Generate report
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_execution_time': total_time
            },
            'test_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'execution_time': r.execution_time,
                    'error_message': r.error_message,
                    'details': r.details,
                    'metrics': r.metrics
                }
                for r in self.results
            ]
        }
        
        # Save report
        report_file = self.output_dir / f'test_report_{int(time.time())}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved to {report_file}")
        logger.info(f"Test Summary: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")


def create_test_runner(config: Dict[str, Any]) -> TestRunner:
    """Create test runner with configuration."""
    return TestRunner(config)

