"""
API Gateway and Service Orchestration System for RL-LLM

This module provides comprehensive API gateway functionality including request routing,
load balancing, rate limiting, service discovery, health monitoring, and microservice
orchestration capabilities for the RL-LLM system.
"""

import asyncio
import aiohttp
from aiohttp import web, ClientSession
import time
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import hashlib
import secrets
from abc import ABC, abstractmethod
import yaml
import consul
import redis
from urllib.parse import urlparse, urljoin
import socket
import subprocess
import psutil
from enum import Enum
import weakref
import pickle
import gzip
import base64

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


@dataclass
class ServiceInstance:
    """Service instance information."""
    service_id: str
    service_name: str
    host: str
    port: int
    protocol: str = "http"
    health_check_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    response_time: float = 0.0
    error_count: int = 0
    request_count: int = 0
    weight: int = 100  # Load balancing weight
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)


@dataclass
class Route:
    """API route configuration."""
    path: str
    service_name: str
    methods: List[str] = field(default_factory=lambda: ["GET"])
    strip_prefix: bool = False
    add_prefix: str = ""
    timeout: int = 30
    retry_count: int = 3
    rate_limit: Optional[int] = None  # requests per minute
    auth_required: bool = False
    roles_required: List[str] = field(default_factory=list)
    middleware: List[str] = field(default_factory=list)
    cache_ttl: Optional[int] = None  # seconds
    circuit_breaker: bool = True


@dataclass
class GatewayConfig:
    """API Gateway configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    default_timeout: int = 30
    health_check_interval: int = 30
    service_discovery_backend: str = "consul"  # consul, redis, static
    load_balancing_strategy: str = "round_robin"  # round_robin, weighted, least_connections
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_caching: bool = True
    cache_backend: str = "redis"
    rate_limiting_backend: str = "redis"
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


class LoadBalancer:
    """Load balancing strategies."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.round_robin_counters = defaultdict(int)
        self.connection_counts = defaultdict(int)
    
    def select_instance(self, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Select service instance based on load balancing strategy."""
        healthy_instances = [i for i in instances if i.status == ServiceStatus.HEALTHY]
        
        if not healthy_instances:
            return None
        
        if self.strategy == "round_robin":
            return self._round_robin_select(healthy_instances)
        elif self.strategy == "weighted":
            return self._weighted_select(healthy_instances)
        elif self.strategy == "least_connections":
            return self._least_connections_select(healthy_instances)
        else:
            return healthy_instances[0]  # Fallback to first instance
    
    def _round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin selection."""
        service_name = instances[0].service_name
        index = self.round_robin_counters[service_name] % len(instances)
        self.round_robin_counters[service_name] += 1
        return instances[index]
    
    def _weighted_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted selection based on instance weights."""
        total_weight = sum(instance.weight for instance in instances)
        if total_weight == 0:
            return instances[0]
        
        import random
        rand_weight = random.randint(1, total_weight)
        
        current_weight = 0
        for instance in instances:
            current_weight += instance.weight
            if rand_weight <= current_weight:
                return instance
        
        return instances[-1]  # Fallback
    
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least connections."""
        return min(instances, key=lambda i: self.connection_counts.get(i.service_id, 0))
    
    def increment_connections(self, service_id: str):
        """Increment connection count for service."""
        self.connection_counts[service_id] += 1
    
    def decrement_connections(self, service_id: str):
        """Decrement connection count for service."""
        if service_id in self.connection_counts:
            self.connection_counts[service_id] = max(0, self.connection_counts[service_id] - 1)


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.timeout:
                self.state = "half_open"
                return True
            return False
        elif self.state == "half_open":
            return True
        
        return False
    
    def record_success(self):
        """Record successful execution."""
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class ServiceDiscovery(ABC):
    """Abstract service discovery interface."""
    
    @abstractmethod
    async def register_service(self, service: ServiceInstance):
        """Register service instance."""
        pass
    
    @abstractmethod
    async def deregister_service(self, service_id: str):
        """Deregister service instance."""
        pass
    
    @abstractmethod
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover service instances."""
        pass
    
    @abstractmethod
    async def get_all_services(self) -> Dict[str, List[ServiceInstance]]:
        """Get all registered services."""
        pass


class ConsulServiceDiscovery(ServiceDiscovery):
    """Consul-based service discovery."""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        self.consul_client = consul.Consul(host=consul_host, port=consul_port)
    
    async def register_service(self, service: ServiceInstance):
        """Register service with Consul."""
        service_def = {
            'ID': service.service_id,
            'Name': service.service_name,
            'Address': service.host,
            'Port': service.port,
            'Tags': service.tags,
            'Meta': service.metadata
        }
        
        if service.health_check_url:
            service_def['Check'] = {
                'HTTP': service.health_check_url,
                'Interval': '30s',
                'Timeout': '10s'
            }
        
        self.consul_client.agent.service.register(**service_def)
        logger.info(f"Registered service {service.service_name} with Consul")
    
    async def deregister_service(self, service_id: str):
        """Deregister service from Consul."""
        self.consul_client.agent.service.deregister(service_id)
        logger.info(f"Deregistered service {service_id} from Consul")
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover services from Consul."""
        _, services = self.consul_client.health.service(service_name, passing=True)
        
        instances = []
        for service_data in services:
            service_info = service_data['Service']
            instances.append(ServiceInstance(
                service_id=service_info['ID'],
                service_name=service_info['Service'],
                host=service_info['Address'],
                port=service_info['Port'],
                metadata=service_info.get('Meta', {}),
                tags=service_info.get('Tags', []),
                status=ServiceStatus.HEALTHY
            ))
        
        return instances
    
    async def get_all_services(self) -> Dict[str, List[ServiceInstance]]:
        """Get all services from Consul."""
        _, services = self.consul_client.catalog.services()
        
        all_services = {}
        for service_name in services.keys():
            all_services[service_name] = await self.discover_services(service_name)
        
        return all_services


class RedisServiceDiscovery(ServiceDiscovery):
    """Redis-based service discovery."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.service_key_prefix = "services:"
    
    async def register_service(self, service: ServiceInstance):
        """Register service with Redis."""
        service_key = f"{self.service_key_prefix}{service.service_name}:{service.service_id}"
        service_data = {
            'service_id': service.service_id,
            'service_name': service.service_name,
            'host': service.host,
            'port': service.port,
            'protocol': service.protocol,
            'health_check_url': service.health_check_url,
            'metadata': json.dumps(service.metadata),
            'status': service.status.value,
            'weight': service.weight,
            'version': service.version,
            'tags': json.dumps(service.tags),
            'registered_at': datetime.now().isoformat()
        }
        
        self.redis_client.hset(service_key, mapping=service_data)
        self.redis_client.expire(service_key, 300)  # 5 minute TTL
        
        # Add to service list
        self.redis_client.sadd(f"{self.service_key_prefix}list:{service.service_name}", service.service_id)
        
        logger.info(f"Registered service {service.service_name} with Redis")
    
    async def deregister_service(self, service_id: str):
        """Deregister service from Redis."""
        # Find and remove from all service lists
        for key in self.redis_client.scan_iter(match=f"{self.service_key_prefix}*:{service_id}"):
            self.redis_client.delete(key)
        
        # Remove from service lists
        for list_key in self.redis_client.scan_iter(match=f"{self.service_key_prefix}list:*"):
            self.redis_client.srem(list_key, service_id)
        
        logger.info(f"Deregistered service {service_id} from Redis")
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover services from Redis."""
        service_ids = self.redis_client.smembers(f"{self.service_key_prefix}list:{service_name}")
        
        instances = []
        for service_id in service_ids:
            service_key = f"{self.service_key_prefix}{service_name}:{service_id}"
            service_data = self.redis_client.hgetall(service_key)
            
            if service_data:
                instances.append(ServiceInstance(
                    service_id=service_data['service_id'],
                    service_name=service_data['service_name'],
                    host=service_data['host'],
                    port=int(service_data['port']),
                    protocol=service_data.get('protocol', 'http'),
                    health_check_url=service_data.get('health_check_url'),
                    metadata=json.loads(service_data.get('metadata', '{}')),
                    status=ServiceStatus(service_data.get('status', 'unknown')),
                    weight=int(service_data.get('weight', 100)),
                    version=service_data.get('version', '1.0.0'),
                    tags=json.loads(service_data.get('tags', '[]'))
                ))
        
        return instances
    
    async def get_all_services(self) -> Dict[str, List[ServiceInstance]]:
        """Get all services from Redis."""
        all_services = {}
        
        for list_key in self.redis_client.scan_iter(match=f"{self.service_key_prefix}list:*"):
            service_name = list_key.split(':')[-1]
            all_services[service_name] = await self.discover_services(service_name)
        
        return all_services


class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, backend: str = "memory", redis_client=None):
        self.backend = backend
        self.redis_client = redis_client
        self.memory_store = defaultdict(deque)
    
    async def is_allowed(self, key: str, limit: int, window: int = 60) -> bool:
        """Check if request is allowed based on rate limit."""
        if self.backend == "redis" and self.redis_client:
            return await self._redis_rate_limit(key, limit, window)
        else:
            return self._memory_rate_limit(key, limit, window)
    
    def _memory_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Memory-based rate limiting."""
        now = time.time()
        
        # Clean old entries
        while self.memory_store[key] and now - self.memory_store[key][0] > window:
            self.memory_store[key].popleft()
        
        # Check limit
        if len(self.memory_store[key]) >= limit:
            return False
        
        # Add current request
        self.memory_store[key].append(now)
        return True
    
    async def _redis_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Redis-based rate limiting using sliding window."""
        now = time.time()
        pipeline = self.redis_client.pipeline()
        
        # Remove old entries
        pipeline.zremrangebyscore(f"rate_limit:{key}", 0, now - window)
        
        # Count current entries
        pipeline.zcard(f"rate_limit:{key}")
        
        # Add current request
        pipeline.zadd(f"rate_limit:{key}", {str(now): now})
        
        # Set expiration
        pipeline.expire(f"rate_limit:{key}", window + 1)
        
        results = pipeline.execute()
        current_count = results[1]
        
        return current_count < limit


class RequestCache:
    """Request caching system."""
    
    def __init__(self, backend: str = "memory", redis_client=None):
        self.backend = backend
        self.redis_client = redis_client
        self.memory_cache = {}
        self.cache_timestamps = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached response."""
        if self.backend == "redis" and self.redis_client:
            cached_data = self.redis_client.get(f"cache:{key}")
            if cached_data:
                return pickle.loads(gzip.decompress(base64.b64decode(cached_data)))
        else:
            if key in self.memory_cache:
                timestamp, data = self.memory_cache[key]
                if time.time() - timestamp < 300:  # 5 minute default TTL
                    return data
                else:
                    del self.memory_cache[key]
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set cached response."""
        if self.backend == "redis" and self.redis_client:
            serialized = base64.b64encode(gzip.compress(pickle.dumps(value))).decode()
            self.redis_client.setex(f"cache:{key}", ttl, serialized)
        else:
            self.memory_cache[key] = (time.time(), value)
    
    def _generate_cache_key(self, request) -> str:
        """Generate cache key for request."""
        key_parts = [
            request.method,
            request.path_qs,
            hashlib.md5(str(sorted(request.headers.items())).encode()).hexdigest()[:8]
        ]
        return ":".join(key_parts)


class APIGateway:
    """Main API Gateway implementation."""
    
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.routes = {}
        self.service_discovery = self._create_service_discovery()
        self.load_balancer = LoadBalancer(config.load_balancing_strategy)
        self.rate_limiter = RateLimiter(config.rate_limiting_backend)
        self.request_cache = RequestCache(config.cache_backend)
        self.circuit_breakers = defaultdict(lambda: CircuitBreaker(
            config.circuit_breaker_threshold, 
            config.circuit_breaker_timeout
        ))
        
        # Metrics
        self.metrics = defaultdict(int)
        self.response_times = defaultdict(list)
        
        # Health monitoring
        self.health_monitor = threading.Thread(target=self._health_monitor_loop, daemon=True)
        self.health_monitor.start()
        
        logger.info("Initialized API Gateway")
    
    def _create_service_discovery(self) -> ServiceDiscovery:
        """Create service discovery backend."""
        if self.config.service_discovery_backend == "consul":
            return ConsulServiceDiscovery()
        elif self.config.service_discovery_backend == "redis":
            return RedisServiceDiscovery()
        else:
            # Static service discovery (for development)
            return StaticServiceDiscovery()
    
    def add_route(self, route: Route):
        """Add API route."""
        self.routes[route.path] = route
        logger.info(f"Added route: {route.path} -> {route.service_name}")
    
    def add_routes_from_config(self, config_path: Path):
        """Add routes from configuration file."""
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml':
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        for route_config in config_data.get('routes', []):
            route = Route(**route_config)
            self.add_route(route)
    
    async def handle_request(self, request: web.Request) -> web.Response:
        """Handle incoming request."""
        start_time = time.time()
        
        try:
            # Find matching route
            route = self._find_route(request.path, request.method)
            if not route:
                return web.Response(status=404, text="Route not found")
            
            # Rate limiting
            if route.rate_limit:
                client_ip = request.remote
                rate_key = f"{client_ip}:{route.path}"
                if not await self.rate_limiter.is_allowed(rate_key, route.rate_limit):
                    return web.Response(status=429, text="Rate limit exceeded")
            
            # Authentication check
            if route.auth_required:
                auth_result = await self._authenticate_request(request)
                if not auth_result:
                    return web.Response(status=401, text="Authentication required")
            
            # Check cache
            if route.cache_ttl and request.method == "GET":
                cache_key = self.request_cache._generate_cache_key(request)
                cached_response = await self.request_cache.get(cache_key)
                if cached_response:
                    self.metrics['cache_hits'] += 1
                    return web.Response(**cached_response)
            
            # Service discovery
            service_instances = await self.service_discovery.discover_services(route.service_name)
            if not service_instances:
                return web.Response(status=503, text="Service unavailable")
            
            # Load balancing
            selected_instance = self.load_balancer.select_instance(service_instances)
            if not selected_instance:
                return web.Response(status=503, text="No healthy service instances")
            
            # Circuit breaker check
            circuit_breaker = self.circuit_breakers[selected_instance.service_id]
            if route.circuit_breaker and not circuit_breaker.can_execute():
                return web.Response(status=503, text="Service temporarily unavailable")
            
            # Forward request
            response = await self._forward_request(request, route, selected_instance)
            
            # Record success
            if route.circuit_breaker:
                circuit_breaker.record_success()
            
            # Cache response
            if route.cache_ttl and request.method == "GET" and response.status == 200:
                cache_data = {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'body': await response.read()
                }
                cache_key = self.request_cache._generate_cache_key(request)
                await self.request_cache.set(cache_key, cache_data, route.cache_ttl)
            
            # Update metrics
            response_time = time.time() - start_time
            self.metrics['requests_total'] += 1
            self.metrics[f'requests_{route.service_name}'] += 1
            self.response_times[route.service_name].append(response_time)
            
            return response
        
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            
            # Record failure for circuit breaker
            if 'route' in locals() and route.circuit_breaker:
                circuit_breaker.record_failure()
            
            self.metrics['errors_total'] += 1
            return web.Response(status=500, text="Internal server error")
    
    def _find_route(self, path: str, method: str) -> Optional[Route]:
        """Find matching route for request."""
        for route_path, route in self.routes.items():
            if self._path_matches(path, route_path) and method in route.methods:
                return route
        return None
    
    def _path_matches(self, request_path: str, route_path: str) -> bool:
        """Check if request path matches route pattern."""
        # Simple path matching - can be enhanced with regex patterns
        if route_path.endswith('*'):
            return request_path.startswith(route_path[:-1])
        return request_path == route_path or request_path.startswith(route_path + '/')
    
    async def _authenticate_request(self, request: web.Request) -> bool:
        """Authenticate request."""
        # Simple token-based authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return False
        
        # In a real implementation, validate the token
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        return len(token) > 0  # Simplified validation
    
    async def _forward_request(self, request: web.Request, route: Route, 
                             instance: ServiceInstance) -> web.Response:
        """Forward request to service instance."""
        # Build target URL
        target_path = request.path
        if route.strip_prefix:
            # Remove route prefix from path
            prefix_to_remove = route.path.rstrip('*')
            if target_path.startswith(prefix_to_remove):
                target_path = target_path[len(prefix_to_remove):]
        
        if route.add_prefix:
            target_path = route.add_prefix + target_path
        
        target_url = f"{instance.protocol}://{instance.host}:{instance.port}{target_path}"
        if request.query_string:
            target_url += f"?{request.query_string}"
        
        # Prepare headers
        headers = dict(request.headers)
        headers.pop('Host', None)  # Remove original host header
        
        # Forward request
        timeout = aiohttp.ClientTimeout(total=route.timeout)
        
        async with ClientSession(timeout=timeout) as session:
            async with session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=await request.read() if request.can_read_body else None
            ) as response:
                
                # Read response
                response_body = await response.read()
                response_headers = dict(response.headers)
                
                # Add CORS headers if enabled
                if self.config.cors_enabled:
                    response_headers.update({
                        'Access-Control-Allow-Origin': ', '.join(self.config.cors_origins),
                        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                    })
                
                return web.Response(
                    status=response.status,
                    headers=response_headers,
                    body=response_body
                )
    
    def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while True:
            try:
                asyncio.run(self._check_service_health())
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(60)
    
    async def _check_service_health(self):
        """Check health of all registered services."""
        all_services = await self.service_discovery.get_all_services()
        
        for service_name, instances in all_services.items():
            for instance in instances:
                if instance.health_check_url:
                    try:
                        timeout = aiohttp.ClientTimeout(total=10)
                        async with ClientSession(timeout=timeout) as session:
                            start_time = time.time()
                            async with session.get(instance.health_check_url) as response:
                                response_time = time.time() - start_time
                                
                                if response.status == 200:
                                    instance.status = ServiceStatus.HEALTHY
                                    instance.response_time = response_time
                                    instance.error_count = 0
                                else:
                                    instance.status = ServiceStatus.UNHEALTHY
                                    instance.error_count += 1
                    
                    except Exception as e:
                        logger.debug(f"Health check failed for {instance.service_id}: {e}")
                        instance.status = ServiceStatus.UNHEALTHY
                        instance.error_count += 1
                    
                    instance.last_health_check = datetime.now()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get gateway metrics."""
        metrics = dict(self.metrics)
        
        # Add response time statistics
        for service_name, times in self.response_times.items():
            if times:
                metrics[f'response_time_avg_{service_name}'] = sum(times) / len(times)
                metrics[f'response_time_max_{service_name}'] = max(times)
                metrics[f'response_time_min_{service_name}'] = min(times)
        
        return metrics
    
    async def start_server(self):
        """Start the API gateway server."""
        app = web.Application()
        
        # Add middleware
        app.middlewares.append(self._logging_middleware)
        app.middlewares.append(self._cors_middleware)
        
        # Add routes
        app.router.add_route('*', '/{path:.*}', self.handle_request)
        
        # Add management endpoints
        app.router.add_get('/health', self._health_endpoint)
        app.router.add_get('/metrics', self._metrics_endpoint)
        app.router.add_get('/services', self._services_endpoint)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()
        
        logger.info(f"API Gateway started on {self.config.host}:{self.config.port}")
    
    async def _logging_middleware(self, request: web.Request, handler):
        """Request logging middleware."""
        start_time = time.time()
        
        try:
            response = await handler(request)
            response_time = time.time() - start_time
            
            logger.info(f"{request.method} {request.path} {response.status} {response_time:.3f}s")
            return response
        
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"{request.method} {request.path} ERROR {response_time:.3f}s: {e}")
            raise
    
    async def _cors_middleware(self, request: web.Request, handler):
        """CORS middleware."""
        if not self.config.cors_enabled:
            return await handler(request)
        
        if request.method == 'OPTIONS':
            return web.Response(
                headers={
                    'Access-Control-Allow-Origin': ', '.join(self.config.cors_origins),
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                }
            )
        
        response = await handler(request)
        response.headers.update({
            'Access-Control-Allow-Origin': ', '.join(self.config.cors_origins),
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'
        })
        
        return response
    
    async def _health_endpoint(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
    
    async def _metrics_endpoint(self, request: web.Request) -> web.Response:
        """Metrics endpoint."""
        return web.json_response(self.get_metrics())
    
    async def _services_endpoint(self, request: web.Request) -> web.Response:
        """Services endpoint."""
        all_services = await self.service_discovery.get_all_services()
        
        services_data = {}
        for service_name, instances in all_services.items():
            services_data[service_name] = [
                {
                    'service_id': instance.service_id,
                    'host': instance.host,
                    'port': instance.port,
                    'status': instance.status.value,
                    'response_time': instance.response_time,
                    'error_count': instance.error_count,
                    'last_health_check': instance.last_health_check.isoformat() if instance.last_health_check else None
                }
                for instance in instances
            ]
        
        return web.json_response(services_data)


class StaticServiceDiscovery(ServiceDiscovery):
    """Static service discovery for development."""
    
    def __init__(self):
        self.services = defaultdict(list)
    
    async def register_service(self, service: ServiceInstance):
        """Register service."""
        self.services[service.service_name].append(service)
        logger.info(f"Registered service {service.service_name} (static)")
    
    async def deregister_service(self, service_id: str):
        """Deregister service."""
        for service_name, instances in self.services.items():
            self.services[service_name] = [i for i in instances if i.service_id != service_id]
        logger.info(f"Deregistered service {service_id} (static)")
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover services."""
        return self.services.get(service_name, [])
    
    async def get_all_services(self) -> Dict[str, List[ServiceInstance]]:
        """Get all services."""
        return dict(self.services)


def create_api_gateway(config: Dict[str, Any]) -> APIGateway:
    """
    Factory function to create API gateway.
    
    Args:
        config: Gateway configuration
        
    Returns:
        APIGateway instance
    """
    gateway_config = GatewayConfig(**config)
    return APIGateway(gateway_config)


async def run_gateway(gateway: APIGateway):
    """Run API gateway server."""
    await gateway.start_server()
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down API Gateway")


if __name__ == "__main__":
    # Example usage
    config = {
        'host': '0.0.0.0',
        'port': 8080,
        'service_discovery_backend': 'static',
        'load_balancing_strategy': 'round_robin'
    }
    
    gateway = create_api_gateway(config)
    
    # Add example routes
    gateway.add_route(Route(
        path='/api/v1/training/*',
        service_name='rl-training-service',
        methods=['GET', 'POST'],
        strip_prefix=True,
        rate_limit=100
    ))
    
    gateway.add_route(Route(
        path='/api/v1/models/*',
        service_name='model-service',
        methods=['GET', 'POST', 'PUT', 'DELETE'],
        strip_prefix=True,
        auth_required=True,
        cache_ttl=300
    ))
    
    # Register example services
    asyncio.run(gateway.service_discovery.register_service(ServiceInstance(
        service_id='rl-training-1',
        service_name='rl-training-service',
        host='localhost',
        port=8001,
        health_check_url='http://localhost:8001/health'
    )))
    
    asyncio.run(gateway.service_discovery.register_service(ServiceInstance(
        service_id='model-service-1',
        service_name='model-service',
        host='localhost',
        port=8002,
        health_check_url='http://localhost:8002/health'
    )))
    
    # Run gateway
    asyncio.run(run_gateway(gateway))

