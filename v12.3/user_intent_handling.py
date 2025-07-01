"""
User Intent Handling System for RL-LLM Tree
Natural language to agent goal translation and context management

This module implements sophisticated intent processing that transforms
natural language instructions into structured RL objectives.
"""

import re
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
from collections import defaultdict, deque
from datetime import datetime

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intents."""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    EXPLORATION = "exploration"
    LEARNING = "learning"
    SOCIAL_INTERACTION = "social_interaction"
    TASK_EXECUTION = "task_execution"
    INFORMATION_GATHERING = "information_gathering"
    CREATIVE = "creative"
    OPTIMIZATION = "optimization"
    SAFETY = "safety"


class UrgencyLevel(Enum):
    """Urgency levels for intent processing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceLevel(Enum):
    """Confidence levels for intent interpretation."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class GoalConstraint:
    """Represents a constraint on goal execution."""
    constraint_type: str
    description: str
    parameters: Dict[str, Any]
    is_hard_constraint: bool = True  # Hard vs soft constraint
    priority: float = 1.0


@dataclass
class SuccessCriterion:
    """Defines success criteria for a goal."""
    criterion_id: str
    description: str
    metric_name: str
    target_value: float
    comparison_operator: str  # ">=", "<=", "==", "!=", ">", "<"
    weight: float = 1.0


@dataclass
class StructuredGoal:
    """Structured representation of a user goal."""
    goal_id: str
    intent_type: IntentType
    primary_objective: str
    sub_objectives: List[str]
    constraints: List[GoalConstraint]
    success_criteria: List[SuccessCriterion]
    urgency: UrgencyLevel
    estimated_duration: Optional[float]  # in seconds
    context: Dict[str, Any]
    dependencies: List[str]  # Other goal IDs this depends on
    created_at: str
    confidence: ConfidenceLevel


@dataclass
class IntentContext:
    """Context information for intent processing."""
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, str]]
    current_environment: Dict[str, Any]
    agent_state: Dict[str, Any]
    user_preferences: Dict[str, Any]
    domain_knowledge: Dict[str, Any]
    timestamp: str


class IntentProcessor(ABC):
    """Abstract base class for intent processors."""
    
    @abstractmethod
    def can_process(self, intent_type: IntentType) -> bool:
        """Check if this processor can handle the given intent type."""
        pass
    
    @abstractmethod
    def process_intent(self, raw_input: str, context: IntentContext) -> StructuredGoal:
        """Process raw input into structured goal."""
        pass


class NavigationIntentProcessor(IntentProcessor):
    """Processes navigation-related intents."""
    
    def __init__(self):
        self.navigation_patterns = [
            (r"go to (.+)", "navigate_to_location"),
            (r"move to (.+)", "navigate_to_location"),
            (r"travel to (.+)", "navigate_to_location"),
            (r"find (.+)", "navigate_to_object"),
            (r"reach (.+)", "navigate_to_location"),
            (r"approach (.+)", "navigate_to_object"),
            (r"avoid (.+)", "navigate_avoiding"),
            (r"follow (.+)", "follow_target"),
            (r"patrol (.+)", "patrol_area"),
            (r"explore (.+)", "explore_area")
        ]
    
    def can_process(self, intent_type: IntentType) -> bool:
        """Check if this processor handles navigation intents."""
        return intent_type == IntentType.NAVIGATION
    
    def process_intent(self, raw_input: str, context: IntentContext) -> StructuredGoal:
        """Process navigation intent."""
        raw_input_lower = raw_input.lower().strip()
        
        # Extract navigation command and target
        navigation_type = "general_navigation"
        target = None
        
        for pattern, nav_type in self.navigation_patterns:
            match = re.search(pattern, raw_input_lower)
            if match:
                navigation_type = nav_type
                target = match.group(1).strip()
                break
        
        # Extract constraints from input
        constraints = self._extract_navigation_constraints(raw_input_lower, context)
        
        # Determine urgency
        urgency = self._determine_urgency(raw_input_lower)
        
        # Create success criteria
        success_criteria = self._create_navigation_success_criteria(navigation_type, target)
        
        # Estimate duration
        estimated_duration = self._estimate_navigation_duration(navigation_type, target, context)
        
        goal = StructuredGoal(
            goal_id=f"nav_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            intent_type=IntentType.NAVIGATION,
            primary_objective=f"Navigate using {navigation_type} to {target}" if target else "Perform navigation task",
            sub_objectives=self._generate_navigation_sub_objectives(navigation_type, target),
            constraints=constraints,
            success_criteria=success_criteria,
            urgency=urgency,
            estimated_duration=estimated_duration,
            context={
                "navigation_type": navigation_type,
                "target": target,
                "original_input": raw_input
            },
            dependencies=[],
            created_at=datetime.now().isoformat(),
            confidence=self._assess_confidence(raw_input, navigation_type, target)
        )
        
        return goal
    
    def _extract_navigation_constraints(self, raw_input: str, context: IntentContext) -> List[GoalConstraint]:
        """Extract navigation constraints from input."""
        constraints = []
        
        # Time constraints
        time_patterns = [
            (r"in (\d+) (second|minute|hour)s?", "time_limit"),
            (r"within (\d+) (second|minute|hour)s?", "time_limit"),
            (r"quickly", "speed_requirement"),
            (r"slowly", "speed_requirement"),
            (r"carefully", "precision_requirement")
        ]
        
        for pattern, constraint_type in time_patterns:
            match = re.search(pattern, raw_input)
            if match:
                if constraint_type == "time_limit":
                    value = int(match.group(1))
                    unit = match.group(2)
                    seconds = value * (60 if unit == "minute" else 3600 if unit == "hour" else 1)
                    constraints.append(GoalConstraint(
                        constraint_type="time_limit",
                        description=f"Complete navigation within {seconds} seconds",
                        parameters={"max_time": seconds},
                        is_hard_constraint=True
                    ))
                elif constraint_type == "speed_requirement":
                    speed_modifier = "fast" if "quickly" in raw_input else "slow"
                    constraints.append(GoalConstraint(
                        constraint_type="speed_requirement",
                        description=f"Navigate at {speed_modifier} speed",
                        parameters={"speed_modifier": speed_modifier},
                        is_hard_constraint=False
                    ))
                elif constraint_type == "precision_requirement":
                    constraints.append(GoalConstraint(
                        constraint_type="precision_requirement",
                        description="Navigate with high precision",
                        parameters={"precision_level": "high"},
                        is_hard_constraint=False
                    ))
        
        # Safety constraints
        if any(word in raw_input for word in ["safe", "safely", "avoid danger"]):
            constraints.append(GoalConstraint(
                constraint_type="safety_requirement",
                description="Maintain safety during navigation",
                parameters={"safety_level": "high"},
                is_hard_constraint=True
            ))
        
        # Path constraints
        if "shortest" in raw_input or "direct" in raw_input:
            constraints.append(GoalConstraint(
                constraint_type="path_optimization",
                description="Use shortest/most direct path",
                parameters={"optimization_type": "shortest_path"},
                is_hard_constraint=False
            ))
        
        return constraints
    
    def _determine_urgency(self, raw_input: str) -> UrgencyLevel:
        """Determine urgency level from input."""
        if any(word in raw_input for word in ["urgent", "emergency", "immediately", "now", "asap"]):
            return UrgencyLevel.CRITICAL
        elif any(word in raw_input for word in ["quickly", "soon", "hurry"]):
            return UrgencyLevel.HIGH
        elif any(word in raw_input for word in ["when convenient", "later", "eventually"]):
            return UrgencyLevel.LOW
        else:
            return UrgencyLevel.MEDIUM
    
    def _create_navigation_success_criteria(self, navigation_type: str, target: Optional[str]) -> List[SuccessCriterion]:
        """Create success criteria for navigation goals."""
        criteria = []
        
        if navigation_type == "navigate_to_location" and target:
            criteria.append(SuccessCriterion(
                criterion_id="reach_target",
                description=f"Reach target location: {target}",
                metric_name="distance_to_target",
                target_value=1.0,  # Within 1 unit of target
                comparison_operator="<=",
                weight=1.0
            ))
        
        # General navigation success criteria
        criteria.append(SuccessCriterion(
            criterion_id="task_completion",
            description="Successfully complete navigation task",
            metric_name="task_completed",
            target_value=1.0,
            comparison_operator="==",
            weight=1.0
        ))
        
        criteria.append(SuccessCriterion(
            criterion_id="collision_avoidance",
            description="Avoid collisions during navigation",
            metric_name="collision_count",
            target_value=0.0,
            comparison_operator="==",
            weight=0.8
        ))
        
        return criteria
    
    def _generate_navigation_sub_objectives(self, navigation_type: str, target: Optional[str]) -> List[str]:
        """Generate sub-objectives for navigation goals."""
        sub_objectives = []
        
        if navigation_type == "navigate_to_location":
            sub_objectives.extend([
                "Plan optimal path to target",
                "Execute movement commands",
                "Monitor progress toward target",
                "Adjust path as needed"
            ])
        elif navigation_type == "explore_area":
            sub_objectives.extend([
                "Identify unexplored regions",
                "Plan exploration strategy",
                "Systematically cover area",
                "Map discovered features"
            ])
        elif navigation_type == "follow_target":
            sub_objectives.extend([
                "Locate target to follow",
                "Maintain appropriate distance",
                "Predict target movement",
                "Adjust following behavior"
            ])
        
        return sub_objectives
    
    def _estimate_navigation_duration(self, navigation_type: str, target: Optional[str], 
                                    context: IntentContext) -> Optional[float]:
        """Estimate duration for navigation task."""
        # Simple heuristic-based estimation
        base_duration = 30.0  # seconds
        
        if navigation_type == "explore_area":
            return base_duration * 3  # Exploration takes longer
        elif navigation_type == "follow_target":
            return None  # Open-ended task
        else:
            return base_duration
    
    def _assess_confidence(self, raw_input: str, navigation_type: str, target: Optional[str]) -> ConfidenceLevel:
        """Assess confidence in intent interpretation."""
        if target and navigation_type != "general_navigation":
            return ConfidenceLevel.HIGH
        elif navigation_type != "general_navigation":
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class TaskExecutionIntentProcessor(IntentProcessor):
    """Processes task execution intents."""
    
    def __init__(self):
        self.task_patterns = [
            (r"complete (.+)", "complete_task"),
            (r"finish (.+)", "complete_task"),
            (r"do (.+)", "execute_task"),
            (r"perform (.+)", "execute_task"),
            (r"accomplish (.+)", "complete_task"),
            (r"achieve (.+)", "achieve_goal"),
            (r"solve (.+)", "solve_problem"),
            (r"fix (.+)", "repair_task"),
            (r"build (.+)", "construction_task"),
            (r"create (.+)", "creation_task")
        ]
    
    def can_process(self, intent_type: IntentType) -> bool:
        """Check if this processor handles task execution intents."""
        return intent_type == IntentType.TASK_EXECUTION
    
    def process_intent(self, raw_input: str, context: IntentContext) -> StructuredGoal:
        """Process task execution intent."""
        raw_input_lower = raw_input.lower().strip()
        
        # Extract task type and description
        task_type = "general_task"
        task_description = raw_input
        
        for pattern, t_type in self.task_patterns:
            match = re.search(pattern, raw_input_lower)
            if match:
                task_type = t_type
                task_description = match.group(1).strip()
                break
        
        # Extract constraints and requirements
        constraints = self._extract_task_constraints(raw_input_lower, context)
        
        # Determine complexity and urgency
        urgency = self._determine_urgency(raw_input_lower)
        complexity = self._assess_task_complexity(task_description, context)
        
        # Create success criteria
        success_criteria = self._create_task_success_criteria(task_type, task_description)
        
        # Generate sub-objectives
        sub_objectives = self._decompose_task(task_type, task_description, complexity)
        
        goal = StructuredGoal(
            goal_id=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            intent_type=IntentType.TASK_EXECUTION,
            primary_objective=f"Execute {task_type}: {task_description}",
            sub_objectives=sub_objectives,
            constraints=constraints,
            success_criteria=success_criteria,
            urgency=urgency,
            estimated_duration=self._estimate_task_duration(complexity, task_type),
            context={
                "task_type": task_type,
                "task_description": task_description,
                "complexity": complexity,
                "original_input": raw_input
            },
            dependencies=self._identify_task_dependencies(task_description, context),
            created_at=datetime.now().isoformat(),
            confidence=self._assess_confidence(raw_input, task_type, task_description)
        )
        
        return goal
    
    def _extract_task_constraints(self, raw_input: str, context: IntentContext) -> List[GoalConstraint]:
        """Extract task-specific constraints."""
        constraints = []
        
        # Quality constraints
        if any(word in raw_input for word in ["perfectly", "precisely", "exactly"]):
            constraints.append(GoalConstraint(
                constraint_type="quality_requirement",
                description="High quality execution required",
                parameters={"quality_level": "high"},
                is_hard_constraint=True
            ))
        
        # Resource constraints
        resource_match = re.search(r"using (.+)", raw_input)
        if resource_match:
            resources = resource_match.group(1).strip()
            constraints.append(GoalConstraint(
                constraint_type="resource_constraint",
                description=f"Use specific resources: {resources}",
                parameters={"required_resources": resources},
                is_hard_constraint=True
            ))
        
        return constraints
    
    def _assess_task_complexity(self, task_description: str, context: IntentContext) -> str:
        """Assess task complexity based on description and context."""
        # Simple heuristic based on description length and keywords
        complex_keywords = ["multiple", "several", "complex", "advanced", "sophisticated"]
        simple_keywords = ["simple", "basic", "easy", "quick"]
        
        if any(keyword in task_description.lower() for keyword in complex_keywords):
            return "high"
        elif any(keyword in task_description.lower() for keyword in simple_keywords):
            return "low"
        elif len(task_description.split()) > 10:
            return "medium"
        else:
            return "low"
    
    def _create_task_success_criteria(self, task_type: str, task_description: str) -> List[SuccessCriterion]:
        """Create success criteria for task execution."""
        criteria = [
            SuccessCriterion(
                criterion_id="task_completion",
                description="Task successfully completed",
                metric_name="task_completed",
                target_value=1.0,
                comparison_operator="==",
                weight=1.0
            )
        ]
        
        if task_type in ["complete_task", "accomplish_task"]:
            criteria.append(SuccessCriterion(
                criterion_id="quality_check",
                description="Task completed with acceptable quality",
                metric_name="quality_score",
                target_value=0.8,
                comparison_operator=">=",
                weight=0.8
            ))
        
        return criteria
    
    def _decompose_task(self, task_type: str, task_description: str, complexity: str) -> List[str]:
        """Decompose task into sub-objectives."""
        base_objectives = [
            "Analyze task requirements",
            "Plan execution strategy",
            "Execute planned actions",
            "Monitor progress",
            "Verify completion"
        ]
        
        if complexity == "high":
            base_objectives.extend([
                "Break down complex components",
                "Coordinate multiple sub-tasks",
                "Handle dependencies",
                "Perform quality assurance"
            ])
        
        return base_objectives
    
    def _estimate_task_duration(self, complexity: str, task_type: str) -> float:
        """Estimate task duration based on complexity and type."""
        base_duration = {"low": 60, "medium": 300, "high": 900}  # seconds
        return base_duration.get(complexity, 300)
    
    def _identify_task_dependencies(self, task_description: str, context: IntentContext) -> List[str]:
        """Identify dependencies for the task."""
        # Simple dependency identification based on keywords
        dependencies = []
        
        if "after" in task_description.lower():
            # Extract dependency from "after X" pattern
            after_match = re.search(r"after (.+)", task_description.lower())
            if after_match:
                dependencies.append(f"prerequisite_{after_match.group(1).strip()}")
        
        return dependencies
    
    def _determine_urgency(self, raw_input: str) -> UrgencyLevel:
        """Determine urgency level from input."""
        if any(word in raw_input for word in ["urgent", "emergency", "immediately"]):
            return UrgencyLevel.CRITICAL
        elif any(word in raw_input for word in ["quickly", "soon", "asap"]):
            return UrgencyLevel.HIGH
        elif any(word in raw_input for word in ["later", "eventually", "when possible"]):
            return UrgencyLevel.LOW
        else:
            return UrgencyLevel.MEDIUM
    
    def _assess_confidence(self, raw_input: str, task_type: str, task_description: str) -> ConfidenceLevel:
        """Assess confidence in intent interpretation."""
        if task_type != "general_task" and len(task_description) > 3:
            return ConfidenceLevel.HIGH
        elif task_type != "general_task":
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class IntentClassifier:
    """Classifies user input into intent types."""
    
    def __init__(self):
        self.intent_keywords = {
            IntentType.NAVIGATION: ["go", "move", "navigate", "travel", "reach", "approach", "find"],
            IntentType.MANIPULATION: ["pick", "grab", "move", "push", "pull", "rotate", "place"],
            IntentType.EXPLORATION: ["explore", "discover", "search", "investigate", "examine"],
            IntentType.LEARNING: ["learn", "practice", "train", "improve", "master"],
            IntentType.SOCIAL_INTERACTION: ["talk", "communicate", "interact", "greet", "ask"],
            IntentType.TASK_EXECUTION: ["do", "perform", "complete", "finish", "accomplish", "execute"],
            IntentType.INFORMATION_GATHERING: ["tell", "explain", "describe", "what", "how", "why"],
            IntentType.CREATIVE: ["create", "build", "design", "make", "generate"],
            IntentType.OPTIMIZATION: ["optimize", "improve", "enhance", "maximize", "minimize"],
            IntentType.SAFETY: ["safe", "danger", "risk", "protect", "secure"]
        }
    
    def classify_intent(self, raw_input: str, context: IntentContext) -> Tuple[IntentType, float]:
        """Classify user input into intent type with confidence score."""
        raw_input_lower = raw_input.lower()
        
        # Score each intent type
        intent_scores = {}
        
        for intent_type, keywords in self.intent_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in raw_input_lower:
                    # Weight by keyword position (earlier = higher weight)
                    position = raw_input_lower.find(keyword)
                    position_weight = 1.0 - (position / len(raw_input_lower))
                    score += position_weight
            
            intent_scores[intent_type] = score
        
        # Find best match
        if not intent_scores or max(intent_scores.values()) == 0:
            return IntentType.TASK_EXECUTION, 0.1  # Default fallback
        
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        confidence = min(best_intent[1], 1.0)  # Cap at 1.0
        
        return best_intent[0], confidence


class UserIntentHandler:
    """Main handler for processing user intents."""
    
    def __init__(self):
        self.classifier = IntentClassifier()
        self.processors: Dict[IntentType, IntentProcessor] = {
            IntentType.NAVIGATION: NavigationIntentProcessor(),
            IntentType.TASK_EXECUTION: TaskExecutionIntentProcessor()
        }
        
        self.intent_history: deque = deque(maxlen=100)
        self.goal_queue: List[StructuredGoal] = []
        self.active_goals: Dict[str, StructuredGoal] = {}
        self.completed_goals: List[StructuredGoal] = []
    
    def process_user_input(self, raw_input: str, context: IntentContext) -> StructuredGoal:
        """Process raw user input into structured goal."""
        # Classify intent
        intent_type, classification_confidence = self.classifier.classify_intent(raw_input, context)
        
        # Get appropriate processor
        processor = self.processors.get(intent_type)
        if not processor:
            # Fallback to task execution processor
            processor = self.processors[IntentType.TASK_EXECUTION]
            intent_type = IntentType.TASK_EXECUTION
        
        # Process intent
        goal = processor.process_intent(raw_input, context)
        
        # Adjust confidence based on classification confidence
        if classification_confidence < 0.5:
            goal.confidence = ConfidenceLevel.LOW
        
        # Store in history
        self.intent_history.append({
            "timestamp": datetime.now().isoformat(),
            "raw_input": raw_input,
            "intent_type": intent_type.value,
            "goal_id": goal.goal_id,
            "classification_confidence": classification_confidence
        })
        
        # Add to goal queue
        self.goal_queue.append(goal)
        
        logger.info(f"Processed intent: {intent_type.value} -> {goal.goal_id}")
        
        return goal
    
    def get_next_goal(self) -> Optional[StructuredGoal]:
        """Get the next goal to execute based on priority."""
        if not self.goal_queue:
            return None
        
        # Sort by urgency and creation time
        urgency_order = {
            UrgencyLevel.CRITICAL: 4,
            UrgencyLevel.HIGH: 3,
            UrgencyLevel.MEDIUM: 2,
            UrgencyLevel.LOW: 1
        }
        
        self.goal_queue.sort(
            key=lambda g: (urgency_order[g.urgency], g.created_at),
            reverse=True
        )
        
        goal = self.goal_queue.pop(0)
        self.active_goals[goal.goal_id] = goal
        
        return goal
    
    def complete_goal(self, goal_id: str, success: bool, performance_metrics: Dict[str, float]):
        """Mark a goal as completed."""
        if goal_id in self.active_goals:
            goal = self.active_goals.pop(goal_id)
            goal.context["completed"] = True
            goal.context["success"] = success
            goal.context["performance_metrics"] = performance_metrics
            goal.context["completion_time"] = datetime.now().isoformat()
            
            self.completed_goals.append(goal)
            
            logger.info(f"Completed goal {goal_id} with success: {success}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of intent handling system."""
        return {
            "queued_goals": len(self.goal_queue),
            "active_goals": len(self.active_goals),
            "completed_goals": len(self.completed_goals),
            "recent_intents": list(self.intent_history)[-5:],
            "goal_queue": [
                {
                    "goal_id": g.goal_id,
                    "intent_type": g.intent_type.value,
                    "urgency": g.urgency.value,
                    "primary_objective": g.primary_objective
                }
                for g in self.goal_queue
            ]
        }
    
    def save_state(self, filepath: str):
        """Save intent handler state to file."""
        state_data = {
            "intent_history": list(self.intent_history),
            "goal_queue": [asdict(goal) for goal in self.goal_queue],
            "active_goals": {gid: asdict(goal) for gid, goal in self.active_goals.items()},
            "completed_goals": [asdict(goal) for goal in self.completed_goals]
        }
        
        # Convert enums to strings for JSON serialization
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj
        
        state_data = convert_enums(state_data)
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)


# Example usage and testing
if __name__ == "__main__":
    # Create intent handler
    handler = UserIntentHandler()
    
    # Create example context
    context = IntentContext(
        user_id="user_001",
        session_id="session_001",
        conversation_history=[],
        current_environment={"location": "office", "objects": ["desk", "chair", "computer"]},
        agent_state={"position": (0, 0), "health": 100},
        user_preferences={"speed": "medium", "safety": "high"},
        domain_knowledge={"known_locations": ["office", "kitchen", "bedroom"]},
        timestamp=datetime.now().isoformat()
    )
    
    # Test various inputs
    test_inputs = [
        "Go to the kitchen quickly",
        "Complete the report by tomorrow",
        "Find the red ball in the living room",
        "Build a tower using blocks",
        "Navigate to the exit safely"
    ]
    
    for input_text in test_inputs:
        print(f"\nProcessing: '{input_text}'")
        goal = handler.process_user_input(input_text, context)
        print(f"Goal ID: {goal.goal_id}")
        print(f"Intent Type: {goal.intent_type.value}")
        print(f"Primary Objective: {goal.primary_objective}")
        print(f"Urgency: {goal.urgency.value}")
        print(f"Confidence: {goal.confidence.value}")
        print(f"Sub-objectives: {goal.sub_objectives[:3]}...")  # First 3
    
    # Get status
    status = handler.get_status()
    print(f"\nHandler Status: {json.dumps(status, indent=2)}")
    
    # Test goal execution
    next_goal = handler.get_next_goal()
    if next_goal:
        print(f"\nNext goal to execute: {next_goal.primary_objective}")
        
        # Simulate completion
        handler.complete_goal(
            next_goal.goal_id, 
            success=True, 
            performance_metrics={"completion_time": 45.0, "efficiency": 0.85}
        )

