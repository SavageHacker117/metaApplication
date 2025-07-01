"""
RL-LLM Integration Bridge for RL-LLM Tree
Core integration layer connecting RL agents with LLM reasoning

This module implements the bridge components that enable seamless
communication between RL agents and LLM systems.
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import deque
import time
from datetime import datetime

# Import our custom components
from curriculum_learning_system import CurriculumManager, SceneConfiguration
from reward_shaping_framework import RewardShaper, ExecutionResult as RewardResult
from user_intent_handling import UserIntentHandler, StructuredGoal, IntentContext
from parallel_execution_framework import ParallelExecutor, ExecutionTask, TaskPriority

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the RL-LLM communication protocol."""
    GOAL_SPECIFICATION = "goal_specification"
    STATE_UPDATE = "state_update"
    ACTION_REQUEST = "action_request"
    ACTION_RESPONSE = "action_response"
    REWARD_FEEDBACK = "reward_feedback"
    LEARNING_UPDATE = "learning_update"
    CONTEXT_UPDATE = "context_update"
    ERROR_REPORT = "error_report"


class AgentState(Enum):
    """States of the RL agent."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    LEARNING = "learning"
    ERROR = "error"


@dataclass
class RLLMMessage:
    """Message structure for RL-LLM communication."""
    message_id: str
    message_type: MessageType
    sender: str
    recipient: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 1
    requires_response: bool = False
    correlation_id: Optional[str] = None


@dataclass
class AgentStatus:
    """Status information for an RL agent."""
    agent_id: str
    state: AgentState
    current_goal: Optional[str]
    current_scene: Optional[str]
    performance_metrics: Dict[str, float]
    learning_progress: Dict[str, float]
    last_action: Optional[str]
    last_reward: float
    episode_count: int
    total_training_time: float
    last_updated: float


class LLMInterface(ABC):
    """Abstract interface for LLM integration."""
    
    @abstractmethod
    async def process_goal(self, goal: StructuredGoal, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a goal and return strategic guidance."""
        pass
    
    @abstractmethod
    async def interpret_state(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Interpret environment state and provide insights."""
        pass
    
    @abstractmethod
    async def suggest_action(self, state: Dict[str, Any], goal: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest high-level actions based on state and goal."""
        pass
    
    @abstractmethod
    async def evaluate_performance(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate agent performance and provide feedback."""
        pass


class MockLLMInterface(LLMInterface):
    """Mock LLM interface for testing and development."""
    
    def __init__(self):
        self.response_delay = 0.1  # Simulate processing time
    
    async def process_goal(self, goal: StructuredGoal, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock goal processing."""
        await asyncio.sleep(self.response_delay)
        
        return {
            "strategic_plan": f"Execute {goal.primary_objective} systematically",
            "key_considerations": [
                "Monitor progress toward success criteria",
                "Adapt strategy based on environmental feedback",
                "Maintain safety constraints"
            ],
            "estimated_difficulty": 0.5,
            "recommended_approach": "incremental_execution",
            "risk_factors": ["time_constraints", "resource_limitations"]
        }
    
    async def interpret_state(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock state interpretation."""
        await asyncio.sleep(self.response_delay)
        
        return {
            "state_summary": "Agent is in a stable position with good visibility",
            "opportunities": ["explore_new_areas", "optimize_current_position"],
            "threats": ["potential_obstacles", "time_pressure"],
            "recommendations": ["maintain_current_strategy", "prepare_for_adaptation"],
            "confidence": 0.8
        }
    
    async def suggest_action(self, state: Dict[str, Any], goal: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock action suggestion."""
        await asyncio.sleep(self.response_delay)
        
        return {
            "suggested_action": "move_forward",
            "action_rationale": "Moving forward aligns with goal progression",
            "alternative_actions": ["turn_left", "turn_right", "wait"],
            "expected_outcome": "progress_toward_goal",
            "confidence": 0.7,
            "risk_level": 0.2
        }
    
    async def evaluate_performance(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock performance evaluation."""
        await asyncio.sleep(self.response_delay)
        
        return {
            "overall_performance": 0.75,
            "strengths": ["efficient_navigation", "goal_achievement"],
            "weaknesses": ["suboptimal_path_planning", "resource_usage"],
            "improvement_suggestions": [
                "Focus on path optimization",
                "Improve resource management",
                "Enhance decision-making speed"
            ],
            "learning_priorities": ["spatial_reasoning", "efficiency_optimization"]
        }


class RLInterface(ABC):
    """Abstract interface for RL agent integration."""
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        pass
    
    @abstractmethod
    def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return result."""
        pass
    
    @abstractmethod
    def get_available_actions(self) -> List[str]:
        """Get list of available actions."""
        pass
    
    @abstractmethod
    def reset_environment(self, scene_config: Optional[SceneConfiguration] = None) -> Dict[str, Any]:
        """Reset environment to initial state."""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        pass


class MockRLInterface(RLInterface):
    """Mock RL interface for testing and development."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = {
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "health": 100.0,
            "energy": 100.0,
            "inventory": [],
            "visible_objects": []
        }
        self.episode_step = 0
        self.total_reward = 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current mock state."""
        return self.state.copy()
    
    def execute_action(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute mock action."""
        self.episode_step += 1
        
        # Simple action simulation
        if action == "move_forward":
            self.state["position"][0] += 1.0
            reward = 1.0
        elif action == "turn_left":
            self.state["orientation"] -= 90.0
            reward = 0.1
        elif action == "turn_right":
            self.state["orientation"] += 90.0
            reward = 0.1
        else:
            reward = 0.0
        
        self.total_reward += reward
        
        return {
            "success": True,
            "reward": reward,
            "new_state": self.get_state(),
            "done": self.episode_step >= 100,  # Episode ends after 100 steps
            "info": {
                "step": self.episode_step,
                "total_reward": self.total_reward
            }
        }
    
    def get_available_actions(self) -> List[str]:
        """Get available mock actions."""
        return ["move_forward", "turn_left", "turn_right", "wait"]
    
    def reset_environment(self, scene_config: Optional[SceneConfiguration] = None) -> Dict[str, Any]:
        """Reset mock environment."""
        self.state = {
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "health": 100.0,
            "energy": 100.0,
            "inventory": [],
            "visible_objects": []
        }
        self.episode_step = 0
        self.total_reward = 0.0
        
        return self.get_state()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get mock performance metrics."""
        return {
            "total_reward": self.total_reward,
            "episode_length": float(self.episode_step),
            "efficiency": self.total_reward / max(self.episode_step, 1),
            "completion_rate": 1.0 if self.episode_step >= 100 else 0.0
        }


class MessageBroker:
    """Handles message routing between RL and LLM components."""
    
    def __init__(self):
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: deque = deque(maxlen=1000)
        self.is_running = False
        self.broker_task: Optional[asyncio.Task] = None
    
    def subscribe(self, message_type: MessageType, callback: Callable):
        """Subscribe to messages of a specific type."""
        if message_type.value not in self.subscribers:
            self.subscribers[message_type.value] = []
        self.subscribers[message_type.value].append(callback)
    
    async def publish(self, message: RLLMMessage):
        """Publish a message to the broker."""
        await self.message_queue.put(message)
        self.message_history.append(asdict(message))
    
    async def start(self):
        """Start the message broker."""
        if self.is_running:
            return
        
        self.is_running = True
        self.broker_task = asyncio.create_task(self._message_loop())
        logger.info("Message broker started")
    
    async def stop(self):
        """Stop the message broker."""
        self.is_running = False
        if self.broker_task:
            self.broker_task.cancel()
            try:
                await self.broker_task
            except asyncio.CancelledError:
                pass
        logger.info("Message broker stopped")
    
    async def _message_loop(self):
        """Main message processing loop."""
        while self.is_running:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                
                # Route message to subscribers
                subscribers = self.subscribers.get(message.message_type.value, [])
                for callback in subscribers:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        logger.error(f"Error in message callback: {e}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in message loop: {e}")


class RLLMBridge:
    """Main bridge component coordinating RL and LLM systems."""
    
    def __init__(self, 
                 llm_interface: LLMInterface,
                 curriculum_manager: CurriculumManager,
                 reward_shaper: RewardShaper,
                 intent_handler: UserIntentHandler,
                 parallel_executor: ParallelExecutor):
        
        self.llm_interface = llm_interface
        self.curriculum_manager = curriculum_manager
        self.reward_shaper = reward_shaper
        self.intent_handler = intent_handler
        self.parallel_executor = parallel_executor
        
        self.message_broker = MessageBroker()
        self.rl_agents: Dict[str, RLInterface] = {}
        self.agent_status: Dict[str, AgentStatus] = {}
        
        # Bridge state
        self.is_running = False
        self.active_episodes: Dict[str, Dict[str, Any]] = {}
        self.episode_history: List[Dict[str, Any]] = []
        
        # Setup message subscriptions
        self._setup_message_handlers()
    
    def _setup_message_handlers(self):
        """Setup message handlers for different message types."""
        self.message_broker.subscribe(MessageType.GOAL_SPECIFICATION, self._handle_goal_specification)
        self.message_broker.subscribe(MessageType.STATE_UPDATE, self._handle_state_update)
        self.message_broker.subscribe(MessageType.ACTION_REQUEST, self._handle_action_request)
        self.message_broker.subscribe(MessageType.REWARD_FEEDBACK, self._handle_reward_feedback)
        self.message_broker.subscribe(MessageType.LEARNING_UPDATE, self._handle_learning_update)
    
    def register_agent(self, agent_id: str, rl_interface: RLInterface):
        """Register an RL agent with the bridge."""
        self.rl_agents[agent_id] = rl_interface
        self.agent_status[agent_id] = AgentStatus(
            agent_id=agent_id,
            state=AgentState.IDLE,
            current_goal=None,
            current_scene=None,
            performance_metrics={},
            learning_progress={},
            last_action=None,
            last_reward=0.0,
            episode_count=0,
            total_training_time=0.0,
            last_updated=time.time()
        )
        logger.info(f"Registered RL agent: {agent_id}")
    
    async def start(self):
        """Start the RL-LLM bridge."""
        if self.is_running:
            return
        
        self.is_running = True
        await self.message_broker.start()
        self.parallel_executor.start()
        
        logger.info("RL-LLM bridge started")
    
    async def stop(self):
        """Stop the RL-LLM bridge."""
        self.is_running = False
        await self.message_broker.stop()
        self.parallel_executor.stop()
        
        logger.info("RL-LLM bridge stopped")
    
    async def process_user_intent(self, user_input: str, context: IntentContext) -> str:
        """Process user intent and initiate agent execution."""
        # Process intent through handler
        goal = self.intent_handler.process_user_input(user_input, context)
        
        # Send goal specification to agents
        message = RLLMMessage(
            message_id=f"goal_{int(time.time() * 1000000)}",
            message_type=MessageType.GOAL_SPECIFICATION,
            sender="intent_handler",
            recipient="all_agents",
            payload={"goal": asdict(goal)},
            requires_response=True
        )
        
        await self.message_broker.publish(message)
        
        return goal.goal_id
    
    async def _handle_goal_specification(self, message: RLLMMessage):
        """Handle goal specification messages."""
        goal_data = message.payload["goal"]
        
        # Process goal through LLM for strategic guidance
        llm_guidance = await self.llm_interface.process_goal(
            StructuredGoal(**goal_data), 
            {"message_context": message.payload}
        )
        
        # Select appropriate agents and scenes
        for agent_id in self.rl_agents.keys():
            # Get next scene from curriculum
            scene_id = self.curriculum_manager.select_next_scene(agent_id)
            if scene_id:
                scene_config = self.curriculum_manager.scenes[scene_id]
                
                # Start episode for agent
                await self._start_episode(agent_id, goal_data, scene_config, llm_guidance)
    
    async def _start_episode(self, agent_id: str, goal_data: Dict[str, Any], 
                           scene_config: SceneConfiguration, llm_guidance: Dict[str, Any]):
        """Start an episode for an agent."""
        agent = self.rl_agents[agent_id]
        status = self.agent_status[agent_id]
        
        # Reset environment with scene configuration
        initial_state = agent.reset_environment(scene_config)
        
        # Update agent status
        status.state = AgentState.PLANNING
        status.current_goal = goal_data["goal_id"]
        status.current_scene = scene_config.scene_id
        status.last_updated = time.time()
        
        # Create episode context
        episode_context = {
            "agent_id": agent_id,
            "goal": goal_data,
            "scene_config": asdict(scene_config),
            "llm_guidance": llm_guidance,
            "initial_state": initial_state,
            "start_time": time.time(),
            "step_count": 0,
            "total_reward": 0.0
        }
        
        self.active_episodes[agent_id] = episode_context
        
        # Start episode execution task
        task = ExecutionTask(
            task_id=f"episode_{agent_id}_{int(time.time())}",
            function=self._execute_episode,
            args=(agent_id,),
            priority=TaskPriority.HIGH
        )
        
        self.parallel_executor.submit_task(task)
        
        logger.info(f"Started episode for agent {agent_id} in scene {scene_config.scene_id}")
    
    async def _execute_episode(self, agent_id: str):
        """Execute an episode for an agent."""
        if agent_id not in self.active_episodes:
            return
        
        episode_context = self.active_episodes[agent_id]
        agent = self.rl_agents[agent_id]
        status = self.agent_status[agent_id]
        
        status.state = AgentState.EXECUTING
        
        try:
            while episode_context["step_count"] < 1000:  # Max episode length
                # Get current state
                current_state = agent.get_state()
                
                # Request action from LLM
                action_guidance = await self.llm_interface.suggest_action(
                    current_state, 
                    episode_context["goal"], 
                    episode_context
                )
                
                # Execute action
                action = action_guidance.get("suggested_action", "wait")
                action_result = agent.execute_action(action, {})
                
                # Calculate reward
                reward_info = {
                    "state": current_state,
                    "action": action,
                    "next_state": action_result["new_state"],
                    "task_completed": action_result.get("done", False),
                    "step_count": episode_context["step_count"]
                }
                
                total_reward, component_rewards = self.reward_shaper.calculate_total_reward(
                    current_state, action, action_result["new_state"], reward_info
                )
                
                # Update episode context
                episode_context["step_count"] += 1
                episode_context["total_reward"] += total_reward
                episode_context["last_action"] = action
                episode_context["last_reward"] = total_reward
                
                # Update agent status
                status.last_action = action
                status.last_reward = total_reward
                status.performance_metrics = agent.get_performance_metrics()
                status.last_updated = time.time()
                
                # Check if episode is done
                if action_result.get("done", False):
                    break
                
                # Brief pause between steps
                await asyncio.sleep(0.01)
            
            # Episode completed
            await self._complete_episode(agent_id)
            
        except Exception as e:
            logger.error(f"Error executing episode for agent {agent_id}: {e}")
            status.state = AgentState.ERROR
            await self._complete_episode(agent_id, success=False)
    
    async def _complete_episode(self, agent_id: str, success: bool = True):
        """Complete an episode and update learning progress."""
        if agent_id not in self.active_episodes:
            return
        
        episode_context = self.active_episodes.pop(agent_id)
        status = self.agent_status[agent_id]
        
        # Update agent status
        status.state = AgentState.LEARNING
        status.episode_count += 1
        status.total_training_time += time.time() - episode_context["start_time"]
        
        # Evaluate performance with LLM
        performance_evaluation = await self.llm_interface.evaluate_performance(episode_context)
        
        # Update curriculum progress
        performance_score = performance_evaluation.get("overall_performance", 0.5)
        self.curriculum_manager.update_progress(
            agent_id, 
            episode_context["scene_config"]["scene_id"],
            performance_score,
            episode_context["step_count"]
        )
        
        # Store episode in history
        episode_summary = {
            "agent_id": agent_id,
            "goal_id": episode_context["goal"]["goal_id"],
            "scene_id": episode_context["scene_config"]["scene_id"],
            "success": success,
            "total_reward": episode_context["total_reward"],
            "step_count": episode_context["step_count"],
            "duration": time.time() - episode_context["start_time"],
            "performance_score": performance_score,
            "llm_evaluation": performance_evaluation
        }
        
        self.episode_history.append(episode_summary)
        
        # Return agent to idle state
        status.state = AgentState.IDLE
        status.current_goal = None
        status.current_scene = None
        status.last_updated = time.time()
        
        logger.info(f"Completed episode for agent {agent_id}: success={success}, score={performance_score}")
    
    async def _handle_state_update(self, message: RLLMMessage):
        """Handle state update messages."""
        # Process state updates from agents
        pass
    
    async def _handle_action_request(self, message: RLLMMessage):
        """Handle action request messages."""
        # Process action requests from agents
        pass
    
    async def _handle_reward_feedback(self, message: RLLMMessage):
        """Handle reward feedback messages."""
        # Process reward feedback
        pass
    
    async def _handle_learning_update(self, message: RLLMMessage):
        """Handle learning update messages."""
        # Process learning updates
        pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "is_running": self.is_running,
            "registered_agents": len(self.rl_agents),
            "active_episodes": len(self.active_episodes),
            "total_episodes": len(self.episode_history),
            "agent_status": {aid: asdict(status) for aid, status in self.agent_status.items()},
            "curriculum_status": {
                aid: self.curriculum_manager.get_curriculum_status(aid) 
                for aid in self.rl_agents.keys()
            },
            "reward_system_status": self.reward_shaper.get_reward_summary(),
            "intent_handler_status": self.intent_handler.get_status(),
            "parallel_executor_status": self.parallel_executor.get_status()
        }


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Create mock components
        llm_interface = MockLLMInterface()
        
        # Create curriculum manager (would need actual config file)
        # curriculum_manager = CurriculumManager("curriculum_config.json")
        
        # Create reward shaper
        from reward_shaping_framework import RewardShaper, AggregationStrategy
        reward_shaper = RewardShaper(AggregationStrategy.WEIGHTED_SUM)
        
        # Create intent handler
        intent_handler = UserIntentHandler()
        
        # Create parallel executor
        parallel_executor = ParallelExecutor()
        
        # For this example, we'll create a minimal curriculum manager
        class MockCurriculumManager:
            def __init__(self):
                self.scenes = {
                    "basic_scene": SceneConfiguration(
                        scene_id="basic_scene",
                        name="Basic Scene",
                        description="Basic training scene",
                        difficulty_level=0.3,
                        difficulty_dimensions={},
                        prerequisites=[],
                        learning_objectives=["basic_navigation"],
                        success_criteria={"completion": 0.8},
                        environment_config={},
                        reward_config={},
                        estimated_training_time=100
                    )
                }
            
            def select_next_scene(self, agent_id: str):
                return "basic_scene"
            
            def update_progress(self, agent_id: str, scene_id: str, performance: float, episodes: int):
                pass
        
        curriculum_manager = MockCurriculumManager()
        
        # Create bridge
        bridge = RLLMBridge(
            llm_interface=llm_interface,
            curriculum_manager=curriculum_manager,
            reward_shaper=reward_shaper,
            intent_handler=intent_handler,
            parallel_executor=parallel_executor
        )
        
        # Register mock agents
        agent1 = MockRLInterface("agent_001")
        bridge.register_agent("agent_001", agent1)
        
        # Start bridge
        await bridge.start()
        
        try:
            # Process user intent
            context = IntentContext(
                user_id="user_001",
                session_id="session_001",
                conversation_history=[],
                current_environment={},
                agent_state={},
                user_preferences={},
                domain_knowledge={},
                timestamp=datetime.now().isoformat()
            )
            
            goal_id = await bridge.process_user_intent("Navigate to the target location", context)
            print(f"Started goal: {goal_id}")
            
            # Wait a bit for execution
            await asyncio.sleep(2.0)
            
            # Get status
            status = bridge.get_system_status()
            print(f"System status: {json.dumps(status, indent=2, default=str)}")
            
        finally:
            await bridge.stop()
    
    # Run the example
    asyncio.run(main())

