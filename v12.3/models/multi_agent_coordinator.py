"""
Multi-Agent Coordination & Swarm Intelligence System for RL-LLM

This module provides comprehensive multi-agent coordination capabilities including
swarm intelligence algorithms, distributed consensus, task allocation, fault-tolerant
communication, emergent behavior simulation, and adaptive coordination protocols.
"""

import asyncio
import time
import json
import logging
import threading
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Callable, Tuple, Union, Set, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import uuid
import random
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import websockets
import aiohttp
from aiohttp import web
import grpc
from grpc import aio as aio_grpc
import redis
import zmq
import zmq.asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import psutil
from abc import ABC, abstractmethod
import hashlib
import pickle
import gzip
import base64
from scipy.spatial.distance import euclidean, cosine
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent operational states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    DISCONNECTED = "disconnected"
    SHUTDOWN = "shutdown"


class MessageType(Enum):
    """Message types for agent communication."""
    HEARTBEAT = "heartbeat"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    COORDINATION = "coordination"
    CONSENSUS = "consensus"
    ELECTION = "election"
    DISCOVERY = "discovery"
    STATUS_UPDATE = "status_update"
    EMERGENCY = "emergency"
    BROADCAST = "broadcast"


class CoordinationStrategy(Enum):
    """Coordination strategies for multi-agent systems."""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HIERARCHICAL = "hierarchical"
    MARKET_BASED = "market_based"
    CONSENSUS_BASED = "consensus_based"
    SWARM_BASED = "swarm_based"


class SwarmAlgorithm(Enum):
    """Swarm intelligence algorithms."""
    PARTICLE_SWARM = "particle_swarm"
    ANT_COLONY = "ant_colony"
    BEE_COLONY = "bee_colony"
    FLOCKING = "flocking"
    FIREFLY = "firefly"
    GREY_WOLF = "grey_wolf"
    WHALE_OPTIMIZATION = "whale_optimization"


@dataclass
class AgentInfo:
    """Agent information and capabilities."""
    agent_id: str
    name: str
    agent_type: str
    capabilities: List[str]
    resources: Dict[str, float]
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    state: AgentState = AgentState.INITIALIZING
    last_seen: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    communication_endpoints: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Task definition for multi-agent execution."""
    task_id: str
    task_type: str
    priority: int
    requirements: Dict[str, Any]
    payload: Dict[str, Any]
    deadline: Optional[datetime] = None
    assigned_agents: List[str] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error_message: str = ""


@dataclass
class Message:
    """Inter-agent communication message."""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: int = 300  # Time to live in seconds
    priority: int = 0
    requires_ack: bool = False
    correlation_id: Optional[str] = None


@dataclass
class SwarmParticle:
    """Particle for swarm optimization algorithms."""
    particle_id: str
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float
    fitness: float = float('inf')
    constraints: Dict[str, Any] = field(default_factory=dict)


class CommunicationProtocol(ABC):
    """Abstract communication protocol interface."""
    
    @abstractmethod
    async def send_message(self, message: Message) -> bool:
        """Send message to recipient(s)."""
        pass
    
    @abstractmethod
    async def receive_messages(self) -> AsyncGenerator[Message, None]:
        """Receive incoming messages."""
        pass
    
    @abstractmethod
    async def broadcast_message(self, message: Message) -> int:
        """Broadcast message to all agents."""
        pass
    
    @abstractmethod
    async def connect(self, endpoint: str) -> bool:
        """Connect to communication network."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from communication network."""
        pass


class WebSocketProtocol(CommunicationProtocol):
    """WebSocket-based communication protocol."""
    
    def __init__(self, agent_id: str, port: int = 8765):
        self.agent_id = agent_id
        self.port = port
        self.connections = {}
        self.server = None
        self.message_queue = asyncio.Queue()
        self.running = False
    
    async def start_server(self):
        """Start WebSocket server."""
        async def handle_client(websocket, path):
            try:
                async for raw_message in websocket:
                    message_data = json.loads(raw_message)
                    message = Message(**message_data)
                    await self.message_queue.put(message)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
        
        self.server = await websockets.serve(handle_client, "localhost", self.port)
        self.running = True
        logger.info(f"WebSocket server started on port {self.port}")
    
    async def connect(self, endpoint: str) -> bool:
        """Connect to another agent's WebSocket."""
        try:
            websocket = await websockets.connect(endpoint)
            agent_id = endpoint.split('/')[-1]  # Extract agent ID from endpoint
            self.connections[agent_id] = websocket
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {endpoint}: {e}")
            return False
    
    async def send_message(self, message: Message) -> bool:
        """Send message via WebSocket."""
        try:
            if message.receiver_id and message.receiver_id in self.connections:
                websocket = self.connections[message.receiver_id]
                message_data = {
                    'message_id': message.message_id,
                    'sender_id': message.sender_id,
                    'receiver_id': message.receiver_id,
                    'message_type': message.message_type.value,
                    'content': message.content,
                    'timestamp': message.timestamp.isoformat(),
                    'ttl': message.ttl,
                    'priority': message.priority,
                    'requires_ack': message.requires_ack,
                    'correlation_id': message.correlation_id
                }
                await websocket.send(json.dumps(message_data))
                return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
        return False
    
    async def receive_messages(self) -> AsyncGenerator[Message, None]:
        """Receive messages from queue."""
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                yield message
            except asyncio.TimeoutError:
                continue
    
    async def broadcast_message(self, message: Message) -> int:
        """Broadcast message to all connected agents."""
        sent_count = 0
        for agent_id, websocket in self.connections.items():
            message.receiver_id = agent_id
            if await self.send_message(message):
                sent_count += 1
        return sent_count
    
    async def disconnect(self) -> bool:
        """Disconnect from all agents."""
        for websocket in self.connections.values():
            await websocket.close()
        self.connections.clear()
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        self.running = False
        return True


class ZeroMQProtocol(CommunicationProtocol):
    """ZeroMQ-based communication protocol."""
    
    def __init__(self, agent_id: str, port: int = 5555):
        self.agent_id = agent_id
        self.port = port
        self.context = zmq.asyncio.Context()
        self.socket = None
        self.running = False
    
    async def connect(self, endpoint: str) -> bool:
        """Connect to ZeroMQ network."""
        try:
            self.socket = self.context.socket(zmq.DEALER)
            self.socket.identity = self.agent_id.encode('utf-8')
            self.socket.connect(endpoint)
            self.running = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ZeroMQ: {e}")
            return False
    
    async def send_message(self, message: Message) -> bool:
        """Send message via ZeroMQ."""
        try:
            if self.socket:
                message_data = pickle.dumps(message)
                await self.socket.send_multipart([
                    message.receiver_id.encode('utf-8') if message.receiver_id else b'',
                    message_data
                ])
                return True
        except Exception as e:
            logger.error(f"Failed to send ZeroMQ message: {e}")
        return False
    
    async def receive_messages(self) -> AsyncGenerator[Message, None]:
        """Receive messages from ZeroMQ."""
        while self.running and self.socket:
            try:
                parts = await self.socket.recv_multipart(zmq.NOBLOCK)
                if len(parts) >= 2:
                    message = pickle.loads(parts[1])
                    yield message
            except zmq.Again:
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"ZeroMQ receive error: {e}")
                break
    
    async def broadcast_message(self, message: Message) -> int:
        """Broadcast message (ZeroMQ doesn't directly support broadcast)."""
        # In a real implementation, this would use a pub-sub pattern
        return await self.send_message(message)
    
    async def disconnect(self) -> bool:
        """Disconnect from ZeroMQ."""
        if self.socket:
            self.socket.close()
        self.context.term()
        self.running = False
        return True


class ConsensusAlgorithm(ABC):
    """Abstract consensus algorithm interface."""
    
    @abstractmethod
    async def propose_value(self, value: Any) -> bool:
        """Propose a value for consensus."""
        pass
    
    @abstractmethod
    async def get_consensus_result(self) -> Optional[Any]:
        """Get the consensus result."""
        pass
    
    @abstractmethod
    async def participate_in_consensus(self, proposal: Any) -> bool:
        """Participate in consensus process."""
        pass


class RaftConsensus(ConsensusAlgorithm):
    """Raft consensus algorithm implementation."""
    
    def __init__(self, agent_id: str, cluster_agents: List[str]):
        self.agent_id = agent_id
        self.cluster_agents = cluster_agents
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.state = "follower"  # follower, candidate, leader
        self.leader_id = None
        self.votes_received = set()
        self.next_index = {}
        self.match_index = {}
        self.commit_index = 0
        self.last_applied = 0
        self.election_timeout = random.uniform(150, 300) / 1000  # 150-300ms
        self.heartbeat_interval = 50 / 1000  # 50ms
        self.last_heartbeat = time.time()
    
    async def start_election(self):
        """Start leader election process."""
        self.state = "candidate"
        self.current_term += 1
        self.voted_for = self.agent_id
        self.votes_received = {self.agent_id}
        self.last_heartbeat = time.time()
        
        # Request votes from other agents
        vote_requests = []
        for agent_id in self.cluster_agents:
            if agent_id != self.agent_id:
                vote_requests.append(self.request_vote(agent_id))
        
        if vote_requests:
            await asyncio.gather(*vote_requests, return_exceptions=True)
        
        # Check if we won the election
        if len(self.votes_received) > len(self.cluster_agents) // 2:
            await self.become_leader()
    
    async def request_vote(self, agent_id: str) -> bool:
        """Request vote from another agent."""
        # In a real implementation, this would send a vote request message
        # For now, simulate a response
        return random.choice([True, False])
    
    async def become_leader(self):
        """Become the cluster leader."""
        self.state = "leader"
        self.leader_id = self.agent_id
        
        # Initialize leader state
        for agent_id in self.cluster_agents:
            if agent_id != self.agent_id:
                self.next_index[agent_id] = len(self.log)
                self.match_index[agent_id] = 0
        
        # Start sending heartbeats
        asyncio.create_task(self.send_heartbeats())
        
        logger.info(f"Agent {self.agent_id} became leader for term {self.current_term}")
    
    async def send_heartbeats(self):
        """Send periodic heartbeats to followers."""
        while self.state == "leader":
            for agent_id in self.cluster_agents:
                if agent_id != self.agent_id:
                    await self.send_append_entries(agent_id)
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def send_append_entries(self, agent_id: str):
        """Send append entries (heartbeat) to follower."""
        # In a real implementation, this would send append entries message
        pass
    
    async def propose_value(self, value: Any) -> bool:
        """Propose a value for consensus."""
        if self.state != "leader":
            return False
        
        # Add to log
        log_entry = {
            'term': self.current_term,
            'value': value,
            'index': len(self.log)
        }
        self.log.append(log_entry)
        
        # Replicate to followers
        replication_tasks = []
        for agent_id in self.cluster_agents:
            if agent_id != self.agent_id:
                replication_tasks.append(self.replicate_to_follower(agent_id, log_entry))
        
        if replication_tasks:
            results = await asyncio.gather(*replication_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            
            # Check if majority accepted
            if success_count >= len(self.cluster_agents) // 2:
                self.commit_index = log_entry['index']
                return True
        
        return False
    
    async def replicate_to_follower(self, agent_id: str, log_entry: Dict[str, Any]) -> bool:
        """Replicate log entry to follower."""
        # In a real implementation, this would send the log entry
        return random.choice([True, False])
    
    async def get_consensus_result(self) -> Optional[Any]:
        """Get the consensus result."""
        if self.commit_index < len(self.log):
            return self.log[self.commit_index]['value']
        return None
    
    async def participate_in_consensus(self, proposal: Any) -> bool:
        """Participate in consensus process."""
        # Followers participate by responding to append entries
        return True


class SwarmOptimizer:
    """Swarm intelligence optimization algorithms."""
    
    def __init__(self, algorithm: SwarmAlgorithm, dimensions: int, 
                 population_size: int = 30, max_iterations: int = 100):
        self.algorithm = algorithm
        self.dimensions = dimensions
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.iteration = 0
        self.fitness_history = []
    
    def initialize_population(self, bounds: List[Tuple[float, float]]):
        """Initialize swarm population."""
        self.particles = []
        
        for i in range(self.population_size):
            position = np.array([
                random.uniform(bounds[j][0], bounds[j][1]) 
                for j in range(self.dimensions)
            ])
            
            velocity = np.array([
                random.uniform(-1, 1) for _ in range(self.dimensions)
            ])
            
            particle = SwarmParticle(
                particle_id=f"particle_{i}",
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=float('inf')
            )
            
            self.particles.append(particle)
    
    def evaluate_fitness(self, objective_function: Callable[[np.ndarray], float]):
        """Evaluate fitness for all particles."""
        for particle in self.particles:
            fitness = objective_function(particle.position)
            particle.fitness = fitness
            
            # Update personal best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            
            # Update global best
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()
    
    def update_particles_pso(self):
        """Update particles using Particle Swarm Optimization."""
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        for particle in self.particles:
            r1 = np.random.random(self.dimensions)
            r2 = np.random.random(self.dimensions)
            
            # Update velocity
            cognitive = c1 * r1 * (particle.best_position - particle.position)
            social = c2 * r2 * (self.global_best_position - particle.position)
            particle.velocity = w * particle.velocity + cognitive + social
            
            # Update position
            particle.position += particle.velocity
    
    def update_particles_aco(self, pheromone_matrix: np.ndarray):
        """Update particles using Ant Colony Optimization."""
        alpha = 1.0  # Pheromone importance
        beta = 2.0   # Heuristic importance
        rho = 0.1    # Evaporation rate
        
        # Update pheromone trails
        pheromone_matrix *= (1 - rho)
        
        for particle in self.particles:
            # Construct solution based on pheromone trails
            for i in range(self.dimensions):
                probabilities = []
                for j in range(len(pheromone_matrix[i])):
                    pheromone = pheromone_matrix[i][j] ** alpha
                    heuristic = (1.0 / (1.0 + abs(particle.position[i] - j))) ** beta
                    probabilities.append(pheromone * heuristic)
                
                # Select next position based on probabilities
                total_prob = sum(probabilities)
                if total_prob > 0:
                    probabilities = [p / total_prob for p in probabilities]
                    particle.position[i] = np.random.choice(
                        len(probabilities), p=probabilities
                    )
    
    def update_particles_flocking(self):
        """Update particles using flocking behavior."""
        separation_radius = 2.0
        alignment_radius = 5.0
        cohesion_radius = 5.0
        
        for particle in self.particles:
            separation = np.zeros(self.dimensions)
            alignment = np.zeros(self.dimensions)
            cohesion = np.zeros(self.dimensions)
            
            neighbors = []
            
            # Find neighbors
            for other in self.particles:
                if other.particle_id != particle.particle_id:
                    distance = np.linalg.norm(particle.position - other.position)
                    
                    if distance < cohesion_radius:
                        neighbors.append(other)
                        
                        # Separation
                        if distance < separation_radius:
                            diff = particle.position - other.position
                            separation += diff / (distance + 1e-6)
                        
                        # Alignment
                        if distance < alignment_radius:
                            alignment += other.velocity
                        
                        # Cohesion
                        cohesion += other.position
            
            if neighbors:
                # Average alignment and cohesion
                alignment /= len(neighbors)
                cohesion = cohesion / len(neighbors) - particle.position
                
                # Update velocity
                particle.velocity += 0.1 * separation + 0.1 * alignment + 0.1 * cohesion
                
                # Limit velocity
                max_velocity = 2.0
                velocity_magnitude = np.linalg.norm(particle.velocity)
                if velocity_magnitude > max_velocity:
                    particle.velocity = particle.velocity / velocity_magnitude * max_velocity
                
                # Update position
                particle.position += particle.velocity
    
    async def optimize(self, objective_function: Callable[[np.ndarray], float],
                     bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        """Run swarm optimization."""
        self.initialize_population(bounds)
        
        for iteration in range(self.max_iterations):
            self.iteration = iteration
            
            # Evaluate fitness
            self.evaluate_fitness(objective_function)
            self.fitness_history.append(self.global_best_fitness)
            
            # Update particles based on algorithm
            if self.algorithm == SwarmAlgorithm.PARTICLE_SWARM:
                self.update_particles_pso()
            elif self.algorithm == SwarmAlgorithm.FLOCKING:
                self.update_particles_flocking()
            elif self.algorithm == SwarmAlgorithm.ANT_COLONY:
                pheromone_matrix = np.ones((self.dimensions, 10))  # Simplified
                self.update_particles_aco(pheromone_matrix)
            
            # Log progress
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Best fitness = {self.global_best_fitness}")
        
        return self.global_best_position, self.global_best_fitness


class TaskAllocator:
    """Task allocation system for multi-agent coordination."""
    
    def __init__(self, strategy: str = "auction"):
        self.strategy = strategy
        self.pending_tasks = {}
        self.active_auctions = {}
        self.task_assignments = {}
    
    async def allocate_task(self, task: Task, available_agents: List[AgentInfo]) -> List[str]:
        """Allocate task to suitable agents."""
        if self.strategy == "auction":
            return await self.auction_based_allocation(task, available_agents)
        elif self.strategy == "greedy":
            return await self.greedy_allocation(task, available_agents)
        elif self.strategy == "optimal":
            return await self.optimal_allocation(task, available_agents)
        else:
            return await self.random_allocation(task, available_agents)
    
    async def auction_based_allocation(self, task: Task, agents: List[AgentInfo]) -> List[str]:
        """Allocate task using auction mechanism."""
        auction_id = str(uuid.uuid4())
        
        # Create auction
        auction = {
            'auction_id': auction_id,
            'task': task,
            'bids': {},
            'deadline': datetime.now() + timedelta(seconds=30)
        }
        
        self.active_auctions[auction_id] = auction
        
        # Request bids from capable agents
        capable_agents = [
            agent for agent in agents
            if self.agent_can_handle_task(agent, task)
        ]
        
        bid_tasks = []
        for agent in capable_agents:
            bid_tasks.append(self.request_bid(agent, task, auction_id))
        
        if bid_tasks:
            await asyncio.gather(*bid_tasks, return_exceptions=True)
        
        # Wait for bids
        await asyncio.sleep(5)  # Wait 5 seconds for bids
        
        # Select winner
        if auction['bids']:
            winner_id = min(auction['bids'].keys(), 
                          key=lambda x: auction['bids'][x]['cost'])
            
            # Clean up auction
            del self.active_auctions[auction_id]
            
            return [winner_id]
        
        return []
    
    async def request_bid(self, agent: AgentInfo, task: Task, auction_id: str):
        """Request bid from agent."""
        # In a real implementation, this would send a bid request message
        # For simulation, generate a random bid
        cost = random.uniform(1.0, 10.0)
        capability_score = self.calculate_capability_score(agent, task)
        
        bid = {
            'agent_id': agent.agent_id,
            'cost': cost,
            'capability_score': capability_score,
            'estimated_completion_time': random.uniform(60, 300)
        }
        
        if auction_id in self.active_auctions:
            self.active_auctions[auction_id]['bids'][agent.agent_id] = bid
    
    def agent_can_handle_task(self, agent: AgentInfo, task: Task) -> bool:
        """Check if agent can handle the task."""
        required_capabilities = task.requirements.get('capabilities', [])
        return all(cap in agent.capabilities for cap in required_capabilities)
    
    def calculate_capability_score(self, agent: AgentInfo, task: Task) -> float:
        """Calculate agent's capability score for the task."""
        score = 0.0
        
        # Check capability match
        required_caps = set(task.requirements.get('capabilities', []))
        agent_caps = set(agent.capabilities)
        
        if required_caps:
            match_ratio = len(required_caps.intersection(agent_caps)) / len(required_caps)
            score += match_ratio * 0.5
        
        # Check resource availability
        required_resources = task.requirements.get('resources', {})
        for resource, amount in required_resources.items():
            if resource in agent.resources:
                if agent.resources[resource] >= amount:
                    score += 0.1
                else:
                    score -= 0.2
        
        # Consider performance metrics
        performance = agent.performance_metrics.get('success_rate', 0.5)
        score += performance * 0.4
        
        return max(0.0, min(1.0, score))
    
    async def greedy_allocation(self, task: Task, agents: List[AgentInfo]) -> List[str]:
        """Greedy task allocation based on capability scores."""
        capable_agents = [
            agent for agent in agents
            if self.agent_can_handle_task(agent, task)
        ]
        
        if not capable_agents:
            return []
        
        # Sort by capability score
        scored_agents = [
            (agent, self.calculate_capability_score(agent, task))
            for agent in capable_agents
        ]
        
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Select top agent(s)
        num_agents_needed = task.requirements.get('num_agents', 1)
        selected = [agent.agent_id for agent, _ in scored_agents[:num_agents_needed]]
        
        return selected
    
    async def optimal_allocation(self, task: Task, agents: List[AgentInfo]) -> List[str]:
        """Optimal task allocation using optimization."""
        capable_agents = [
            agent for agent in agents
            if self.agent_can_handle_task(agent, task)
        ]
        
        if not capable_agents:
            return []
        
        # This is a simplified version - in practice, you'd use more sophisticated optimization
        num_agents_needed = task.requirements.get('num_agents', 1)
        
        if len(capable_agents) <= num_agents_needed:
            return [agent.agent_id for agent in capable_agents]
        
        # Use capability scores for selection
        scored_agents = [
            (agent, self.calculate_capability_score(agent, task))
            for agent in capable_agents
        ]
        
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return [agent.agent_id for agent, _ in scored_agents[:num_agents_needed]]
    
    async def random_allocation(self, task: Task, agents: List[AgentInfo]) -> List[str]:
        """Random task allocation."""
        capable_agents = [
            agent for agent in agents
            if self.agent_can_handle_task(agent, task)
        ]
        
        if not capable_agents:
            return []
        
        num_agents_needed = task.requirements.get('num_agents', 1)
        selected = random.sample(capable_agents, 
                                min(num_agents_needed, len(capable_agents)))
        
        return [agent.agent_id for agent in selected]


class MultiAgentCoordinator:
    """Main multi-agent coordination system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = config.get('agent_id', str(uuid.uuid4()))
        self.agents = {}
        self.tasks = {}
        self.messages = deque(maxlen=10000)
        
        # Communication
        protocol_type = config.get('communication_protocol', 'websocket')
        if protocol_type == 'websocket':
            self.communication = WebSocketProtocol(
                self.agent_id, 
                config.get('communication_port', 8765)
            )
        elif protocol_type == 'zeromq':
            self.communication = ZeroMQProtocol(
                self.agent_id,
                config.get('communication_port', 5555)
            )
        
        # Coordination components
        self.consensus = RaftConsensus(self.agent_id, config.get('cluster_agents', []))
        self.task_allocator = TaskAllocator(config.get('allocation_strategy', 'auction'))
        self.swarm_optimizer = SwarmOptimizer(
            SwarmAlgorithm.PARTICLE_SWARM,
            config.get('optimization_dimensions', 10)
        )
        
        # State management
        self.state = AgentState.INITIALIZING
        self.coordination_strategy = CoordinationStrategy(
            config.get('coordination_strategy', 'decentralized')
        )
        
        # Performance tracking
        self.performance_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'coordination_events': 0
        }
        
        # Background tasks
        self.background_tasks = []
        self.running = False
        
        logger.info(f"Initialized MultiAgentCoordinator {self.agent_id}")
    
    async def start(self):
        """Start the coordination system."""
        self.running = True
        self.state = AgentState.ACTIVE
        
        # Start communication
        if hasattr(self.communication, 'start_server'):
            await self.communication.start_server()
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self.message_handler()),
            asyncio.create_task(self.heartbeat_sender()),
            asyncio.create_task(self.task_monitor()),
            asyncio.create_task(self.performance_monitor())
        ]
        
        logger.info(f"MultiAgentCoordinator {self.agent_id} started")
    
    async def stop(self):
        """Stop the coordination system."""
        self.running = False
        self.state = AgentState.SHUTDOWN
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Disconnect communication
        await self.communication.disconnect()
        
        logger.info(f"MultiAgentCoordinator {self.agent_id} stopped")
    
    async def register_agent(self, agent_info: AgentInfo):
        """Register a new agent in the system."""
        self.agents[agent_info.agent_id] = agent_info
        
        # Send discovery message
        discovery_message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=None,  # Broadcast
            message_type=MessageType.DISCOVERY,
            content={
                'action': 'agent_joined',
                'agent_info': {
                    'agent_id': agent_info.agent_id,
                    'name': agent_info.name,
                    'agent_type': agent_info.agent_type,
                    'capabilities': agent_info.capabilities,
                    'resources': agent_info.resources
                }
            }
        )
        
        await self.broadcast_message(discovery_message)
        logger.info(f"Registered agent {agent_info.agent_id}")
    
    async def submit_task(self, task: Task) -> bool:
        """Submit a task for execution."""
        self.tasks[task.task_id] = task
        
        # Allocate task to agents
        available_agents = [
            agent for agent in self.agents.values()
            if agent.state == AgentState.ACTIVE
        ]
        
        assigned_agents = await self.task_allocator.allocate_task(task, available_agents)
        
        if assigned_agents:
            task.assigned_agents = assigned_agents
            task.status = "assigned"
            
            # Send task requests
            for agent_id in assigned_agents:
                task_message = Message(
                    message_id=str(uuid.uuid4()),
                    sender_id=self.agent_id,
                    receiver_id=agent_id,
                    message_type=MessageType.TASK_REQUEST,
                    content={
                        'task_id': task.task_id,
                        'task_type': task.task_type,
                        'priority': task.priority,
                        'requirements': task.requirements,
                        'payload': task.payload,
                        'deadline': task.deadline.isoformat() if task.deadline else None
                    }
                )
                
                await self.send_message(task_message)
            
            logger.info(f"Task {task.task_id} assigned to agents: {assigned_agents}")
            return True
        else:
            task.status = "failed"
            task.error_message = "No suitable agents available"
            logger.warning(f"Failed to assign task {task.task_id}")
            return False
    
    async def send_message(self, message: Message) -> bool:
        """Send message to another agent."""
        success = await self.communication.send_message(message)
        if success:
            self.performance_metrics['messages_sent'] += 1
            self.messages.append(message)
        return success
    
    async def broadcast_message(self, message: Message) -> int:
        """Broadcast message to all agents."""
        sent_count = await self.communication.broadcast_message(message)
        self.performance_metrics['messages_sent'] += sent_count
        if sent_count > 0:
            self.messages.append(message)
        return sent_count
    
    async def message_handler(self):
        """Handle incoming messages."""
        async for message in self.communication.receive_messages():
            try:
                self.performance_metrics['messages_received'] += 1
                self.messages.append(message)
                
                await self.process_message(message)
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def process_message(self, message: Message):
        """Process incoming message."""
        if message.message_type == MessageType.HEARTBEAT:
            await self.handle_heartbeat(message)
        elif message.message_type == MessageType.TASK_REQUEST:
            await self.handle_task_request(message)
        elif message.message_type == MessageType.TASK_RESPONSE:
            await self.handle_task_response(message)
        elif message.message_type == MessageType.COORDINATION:
            await self.handle_coordination_message(message)
        elif message.message_type == MessageType.CONSENSUS:
            await self.handle_consensus_message(message)
        elif message.message_type == MessageType.DISCOVERY:
            await self.handle_discovery_message(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
            await self.handle_status_update(message)
        elif message.message_type == MessageType.EMERGENCY:
            await self.handle_emergency_message(message)
    
    async def handle_heartbeat(self, message: Message):
        """Handle heartbeat message."""
        sender_id = message.sender_id
        if sender_id in self.agents:
            self.agents[sender_id].last_seen = datetime.now()
            self.agents[sender_id].state = AgentState.ACTIVE
    
    async def handle_task_request(self, message: Message):
        """Handle task request message."""
        content = message.content
        task_id = content['task_id']
        
        # Create task from message
        task = Task(
            task_id=task_id,
            task_type=content['task_type'],
            priority=content['priority'],
            requirements=content['requirements'],
            payload=content['payload'],
            deadline=datetime.fromisoformat(content['deadline']) if content.get('deadline') else None
        )
        
        # Process task (simplified - in practice, this would involve actual task execution)
        await asyncio.sleep(random.uniform(1, 5))  # Simulate task execution
        
        # Send response
        response = Message(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            message_type=MessageType.TASK_RESPONSE,
            content={
                'task_id': task_id,
                'status': 'completed',
                'result': {'success': True, 'data': 'task_result'},
                'execution_time': random.uniform(1, 5)
            },
            correlation_id=message.message_id
        )
        
        await self.send_message(response)
    
    async def handle_task_response(self, message: Message):
        """Handle task response message."""
        content = message.content
        task_id = content['task_id']
        
        if task_id in self.tasks:
            task = self.tasks[task_id]
            
            if content['status'] == 'completed':
                task.status = 'completed'
                task.result = content.get('result')
                task.completed_at = datetime.now()
                self.performance_metrics['tasks_completed'] += 1
            else:
                task.status = 'failed'
                task.error_message = content.get('error_message', 'Task failed')
                self.performance_metrics['tasks_failed'] += 1
            
            logger.info(f"Task {task_id} {task.status}")
    
    async def handle_coordination_message(self, message: Message):
        """Handle coordination message."""
        self.performance_metrics['coordination_events'] += 1
        # Process coordination logic based on strategy
        pass
    
    async def handle_consensus_message(self, message: Message):
        """Handle consensus message."""
        # Participate in consensus process
        await self.consensus.participate_in_consensus(message.content)
    
    async def handle_discovery_message(self, message: Message):
        """Handle discovery message."""
        content = message.content
        
        if content['action'] == 'agent_joined':
            agent_info_data = content['agent_info']
            agent_info = AgentInfo(
                agent_id=agent_info_data['agent_id'],
                name=agent_info_data['name'],
                agent_type=agent_info_data['agent_type'],
                capabilities=agent_info_data['capabilities'],
                resources=agent_info_data['resources']
            )
            
            if agent_info.agent_id not in self.agents:
                self.agents[agent_info.agent_id] = agent_info
                logger.info(f"Discovered new agent: {agent_info.agent_id}")
    
    async def handle_status_update(self, message: Message):
        """Handle status update message."""
        sender_id = message.sender_id
        if sender_id in self.agents:
            content = message.content
            agent = self.agents[sender_id]
            
            # Update agent status
            if 'state' in content:
                agent.state = AgentState(content['state'])
            if 'resources' in content:
                agent.resources.update(content['resources'])
            if 'performance_metrics' in content:
                agent.performance_metrics.update(content['performance_metrics'])
    
    async def handle_emergency_message(self, message: Message):
        """Handle emergency message."""
        logger.warning(f"Emergency message from {message.sender_id}: {message.content}")
        
        # Implement emergency response logic
        if message.content.get('type') == 'agent_failure':
            failed_agent_id = message.content.get('failed_agent_id')
            if failed_agent_id in self.agents:
                self.agents[failed_agent_id].state = AgentState.FAILED
                
                # Reassign tasks from failed agent
                await self.handle_agent_failure(failed_agent_id)
    
    async def handle_agent_failure(self, failed_agent_id: str):
        """Handle agent failure by reassigning tasks."""
        # Find tasks assigned to failed agent
        affected_tasks = [
            task for task in self.tasks.values()
            if failed_agent_id in task.assigned_agents and task.status in ['assigned', 'running']
        ]
        
        for task in affected_tasks:
            # Remove failed agent from assignment
            task.assigned_agents.remove(failed_agent_id)
            
            # Try to reassign to other agents
            available_agents = [
                agent for agent in self.agents.values()
                if agent.state == AgentState.ACTIVE and agent.agent_id != failed_agent_id
            ]
            
            new_assignments = await self.task_allocator.allocate_task(task, available_agents)
            
            if new_assignments:
                task.assigned_agents.extend(new_assignments)
                logger.info(f"Reassigned task {task.task_id} to {new_assignments}")
            else:
                task.status = 'failed'
                task.error_message = 'No agents available for reassignment'
                logger.warning(f"Failed to reassign task {task.task_id}")
    
    async def heartbeat_sender(self):
        """Send periodic heartbeats."""
        while self.running:
            heartbeat = Message(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                receiver_id=None,  # Broadcast
                message_type=MessageType.HEARTBEAT,
                content={
                    'timestamp': datetime.now().isoformat(),
                    'state': self.state.value,
                    'performance_metrics': self.performance_metrics
                }
            )
            
            await self.broadcast_message(heartbeat)
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
    
    async def task_monitor(self):
        """Monitor task execution and handle timeouts."""
        while self.running:
            current_time = datetime.now()
            
            for task in self.tasks.values():
                if task.status == 'assigned' and task.deadline:
                    if current_time > task.deadline:
                        task.status = 'timeout'
                        task.error_message = 'Task deadline exceeded'
                        self.performance_metrics['tasks_failed'] += 1
                        logger.warning(f"Task {task.task_id} timed out")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def performance_monitor(self):
        """Monitor system performance."""
        while self.running:
            # Log performance metrics
            active_agents = sum(1 for agent in self.agents.values() 
                              if agent.state == AgentState.ACTIVE)
            
            logger.info(f"Performance: {active_agents} active agents, "
                       f"{len(self.tasks)} total tasks, "
                       f"{self.performance_metrics['tasks_completed']} completed, "
                       f"{self.performance_metrics['tasks_failed']} failed")
            
            await asyncio.sleep(300)  # Log every 5 minutes
    
    async def optimize_coordination(self, objective_function: Callable[[np.ndarray], float]):
        """Optimize coordination parameters using swarm intelligence."""
        bounds = [(0.0, 1.0) for _ in range(self.config.get('optimization_dimensions', 10))]
        
        best_params, best_fitness = await self.swarm_optimizer.optimize(
            objective_function, bounds
        )
        
        logger.info(f"Optimization completed: best fitness = {best_fitness}")
        return best_params, best_fitness
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        active_agents = [agent for agent in self.agents.values() 
                        if agent.state == AgentState.ACTIVE]
        
        task_status = defaultdict(int)
        for task in self.tasks.values():
            task_status[task.status] += 1
        
        return {
            'coordinator_id': self.agent_id,
            'state': self.state.value,
            'total_agents': len(self.agents),
            'active_agents': len(active_agents),
            'total_tasks': len(self.tasks),
            'task_status': dict(task_status),
            'performance_metrics': self.performance_metrics,
            'coordination_strategy': self.coordination_strategy.value,
            'uptime': time.time() - self.config.get('start_time', time.time())
        }
    
    def visualize_network(self, output_path: Optional[Path] = None):
        """Visualize the agent network."""
        G = nx.Graph()
        
        # Add nodes for agents
        for agent in self.agents.values():
            G.add_node(agent.agent_id, 
                      agent_type=agent.agent_type,
                      state=agent.state.value)
        
        # Add edges based on communication (simplified)
        agent_ids = list(self.agents.keys())
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                # In a real implementation, this would be based on actual communication patterns
                if random.random() > 0.7:  # 30% chance of connection
                    G.add_edge(agent_ids[i], agent_ids[j])
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Position nodes
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Color nodes by state
        node_colors = []
        for node in G.nodes():
            agent = self.agents[node]
            if agent.state == AgentState.ACTIVE:
                node_colors.append('green')
            elif agent.state == AgentState.BUSY:
                node_colors.append('orange')
            elif agent.state == AgentState.FAILED:
                node_colors.append('red')
            else:
                node_colors.append('gray')
        
        # Draw network
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=1000,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                alpha=0.7)
        
        plt.title("Multi-Agent Network Topology")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def create_multi_agent_coordinator(config: Dict[str, Any]) -> MultiAgentCoordinator:
    """
    Factory function to create multi-agent coordinator.
    
    Args:
        config: Coordinator configuration
        
    Returns:
        MultiAgentCoordinator instance
    """
    config['start_time'] = time.time()
    return MultiAgentCoordinator(config)


# Example usage and testing
async def example_usage():
    """Example usage of multi-agent coordination system."""
    
    # Configuration
    config = {
        'agent_id': 'coordinator_1',
        'communication_protocol': 'websocket',
        'communication_port': 8765,
        'coordination_strategy': 'decentralized',
        'allocation_strategy': 'auction',
        'optimization_dimensions': 5,
        'cluster_agents': ['agent_1', 'agent_2', 'agent_3']
    }
    
    # Create coordinator
    coordinator = create_multi_agent_coordinator(config)
    
    # Start coordinator
    await coordinator.start()
    
    # Register some agents
    agents = [
        AgentInfo(
            agent_id=f"agent_{i}",
            name=f"Agent {i}",
            agent_type="worker",
            capabilities=["compute", "storage"] if i % 2 == 0 else ["network", "analysis"],
            resources={"cpu": random.uniform(0.5, 1.0), "memory": random.uniform(0.3, 0.8)}
        )
        for i in range(5)
    ]
    
    for agent in agents:
        await coordinator.register_agent(agent)
    
    # Submit some tasks
    tasks = [
        Task(
            task_id=f"task_{i}",
            task_type="computation",
            priority=random.randint(1, 10),
            requirements={
                "capabilities": ["compute"],
                "resources": {"cpu": 0.2, "memory": 0.1},
                "num_agents": 1
            },
            payload={"data": f"task_data_{i}"}
        )
        for i in range(10)
    ]
    
    for task in tasks:
        await coordinator.submit_task(task)
    
    # Run for a while
    await asyncio.sleep(60)
    
    # Get status
    status = coordinator.get_system_status()
    print(f"System Status: {json.dumps(status, indent=2)}")
    
    # Visualize network
    coordinator.visualize_network()
    
    # Stop coordinator
    await coordinator.stop()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())

