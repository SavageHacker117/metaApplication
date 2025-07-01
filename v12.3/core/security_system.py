"""
Security and Monitoring System for RL-LLM

This module provides comprehensive security and monitoring capabilities including
authentication, authorization, audit logging, intrusion detection, system monitoring,
and security compliance features for the RL-LLM system.
"""

import os
import time
import hashlib
import secrets
import jwt
import bcrypt
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import sqlite3
import json
import psutil
import socket
import subprocess
from abc import ABC, abstractmethod
import re
import ipaddress
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import yaml

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_expiration_hours: int = 24
    password_min_length: int = 8
    password_require_special: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    session_timeout_minutes: int = 120
    enable_2fa: bool = False
    audit_log_retention_days: int = 90
    encryption_key: Optional[str] = None
    allowed_ip_ranges: List[str] = field(default_factory=list)
    rate_limit_requests_per_minute: int = 60
    enable_intrusion_detection: bool = True
    security_headers: bool = True
    data_classification_levels: List[str] = field(default_factory=lambda: ['public', 'internal', 'confidential', 'restricted'])


@dataclass
class User:
    """User data structure."""
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: List[str]
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    two_factor_secret: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str
    timestamp: datetime
    user_id: Optional[str]
    event_type: str
    resource: str
    action: str
    result: str  # 'success', 'failure', 'error'
    ip_address: str
    user_agent: str
    details: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"  # 'low', 'medium', 'high', 'critical'


@dataclass
class SecurityAlert:
    """Security alert data structure."""
    alert_id: str
    timestamp: datetime
    alert_type: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    title: str
    description: str
    source_ip: Optional[str] = None
    affected_user: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class AuthenticationManager:
    """Authentication and user management."""
    
    def __init__(self, config: SecurityConfig, db_path: Path):
        self.config = config
        self.db_path = db_path
        self._init_database()
        
        # Rate limiting
        self.login_attempts = defaultdict(deque)
        self.rate_limits = defaultdict(deque)
        
        # Encryption
        if config.encryption_key:
            self.cipher = Fernet(config.encryption_key.encode())
        else:
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            logger.warning("Generated new encryption key. Store it securely!")
        
        logger.info("Initialized AuthenticationManager")
    
    def _init_database(self):
        """Initialize user database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    roles TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TEXT,
                    two_factor_secret TEXT,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            conn.commit()
    
    def create_user(self, username: str, email: str, password: str, 
                   roles: List[str] = None, permissions: List[str] = None) -> str:
        """Create new user."""
        # Validate password
        if not self._validate_password(password):
            raise ValueError("Password does not meet security requirements")
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        user = User(
            user_id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or ['user'],
            permissions=permissions or [],
            created_at=datetime.now()
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users 
                (user_id, username, email, password_hash, roles, permissions, 
                 created_at, is_active, failed_login_attempts, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user.user_id, user.username, user.email, user.password_hash,
                json.dumps(user.roles), json.dumps(user.permissions),
                user.created_at.isoformat(), user.is_active,
                user.failed_login_attempts, json.dumps(user.metadata)
            ))
            
            conn.commit()
        
        logger.info(f"Created user: {username}")
        return user.user_id
    
    def authenticate_user(self, username: str, password: str, ip_address: str = "") -> Optional[str]:
        """Authenticate user and return JWT token."""
        # Check rate limiting
        if not self._check_rate_limit(ip_address, 'login'):
            raise ValueError("Too many login attempts. Please try again later.")
        
        # Get user from database
        user = self._get_user_by_username(username)
        if not user:
            self._record_failed_login(username, ip_address)
            return None
        
        # Check if user is locked
        if user.locked_until and datetime.now() < user.locked_until:
            raise ValueError("Account is temporarily locked due to too many failed attempts")
        
        # Verify password
        if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            self._record_failed_login(username, ip_address)
            return None
        
        # Reset failed attempts on successful login
        self._reset_failed_attempts(user.user_id)
        
        # Update last login
        self._update_last_login(user.user_id)
        
        # Generate JWT token
        token = self._generate_jwt_token(user)
        
        # Create session
        session_id = self._create_session(user.user_id, ip_address)
        
        logger.info(f"User authenticated: {username}")
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return user info."""
        try:
            payload = jwt.decode(token, self.config.secret_key, algorithms=['HS256'])
            
            # Check if user still exists and is active
            user = self._get_user_by_id(payload['user_id'])
            if not user or not user.is_active:
                return None
            
            return payload
        
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def _validate_password(self, password: str) -> bool:
        """Validate password against security policy."""
        if len(password) < self.config.password_min_length:
            return False
        
        if self.config.password_require_special:
            if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                return False
            if not re.search(r'[A-Z]', password):
                return False
            if not re.search(r'[a-z]', password):
                return False
            if not re.search(r'\d', password):
                return False
        
        return True
    
    def _check_rate_limit(self, identifier: str, action: str) -> bool:
        """Check rate limiting."""
        now = time.time()
        key = f"{identifier}_{action}"
        
        # Clean old attempts
        while (self.rate_limits[key] and 
               now - self.rate_limits[key][0] > 60):  # 1 minute window
            self.rate_limits[key].popleft()
        
        # Check limit
        if len(self.rate_limits[key]) >= self.config.rate_limit_requests_per_minute:
            return False
        
        # Record attempt
        self.rate_limits[key].append(now)
        return True
    
    def _record_failed_login(self, username: str, ip_address: str):
        """Record failed login attempt."""
        user = self._get_user_by_username(username)
        if not user:
            return
        
        failed_attempts = user.failed_login_attempts + 1
        
        # Lock account if too many failures
        locked_until = None
        if failed_attempts >= self.config.max_login_attempts:
            locked_until = datetime.now() + timedelta(minutes=self.config.lockout_duration_minutes)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users 
                SET failed_login_attempts = ?, locked_until = ?
                WHERE user_id = ?
            ''', (failed_attempts, locked_until.isoformat() if locked_until else None, user.user_id))
            conn.commit()
        
        logger.warning(f"Failed login attempt for {username} from {ip_address}")
    
    def _reset_failed_attempts(self, user_id: str):
        """Reset failed login attempts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users 
                SET failed_login_attempts = 0, locked_until = NULL
                WHERE user_id = ?
            ''', (user_id,))
            conn.commit()
    
    def _update_last_login(self, user_id: str):
        """Update last login timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users 
                SET last_login = ?
                WHERE user_id = ?
            ''', (datetime.now().isoformat(), user_id))
            conn.commit()
    
    def _generate_jwt_token(self, user: User) -> str:
        """Generate JWT token for user."""
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'roles': user.roles,
            'permissions': user.permissions,
            'exp': datetime.utcnow() + timedelta(hours=self.config.jwt_expiration_hours),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.config.secret_key, algorithm='HS256')
    
    def _create_session(self, user_id: str, ip_address: str) -> str:
        """Create user session."""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(minutes=self.config.session_timeout_minutes)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions 
                (session_id, user_id, created_at, expires_at, ip_address, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, user_id, datetime.now().isoformat(), 
                  expires_at.isoformat(), ip_address, True))
            conn.commit()
        
        return session_id
    
    def _get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            columns = [desc[0] for desc in cursor.description]
            data = dict(zip(columns, row))
            
            return User(
                user_id=data['user_id'],
                username=data['username'],
                email=data['email'],
                password_hash=data['password_hash'],
                roles=json.loads(data['roles']),
                permissions=json.loads(data['permissions']),
                created_at=datetime.fromisoformat(data['created_at']),
                last_login=datetime.fromisoformat(data['last_login']) if data['last_login'] else None,
                is_active=data['is_active'],
                failed_login_attempts=data['failed_login_attempts'],
                locked_until=datetime.fromisoformat(data['locked_until']) if data['locked_until'] else None,
                two_factor_secret=data['two_factor_secret'],
                metadata=json.loads(data['metadata']) if data['metadata'] else {}
            )
    
    def _get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            columns = [desc[0] for desc in cursor.description]
            data = dict(zip(columns, row))
            
            return User(
                user_id=data['user_id'],
                username=data['username'],
                email=data['email'],
                password_hash=data['password_hash'],
                roles=json.loads(data['roles']),
                permissions=json.loads(data['permissions']),
                created_at=datetime.fromisoformat(data['created_at']),
                last_login=datetime.fromisoformat(data['last_login']) if data['last_login'] else None,
                is_active=data['is_active'],
                failed_login_attempts=data['failed_login_attempts'],
                locked_until=datetime.fromisoformat(data['locked_until']) if data['locked_until'] else None,
                two_factor_secret=data['two_factor_secret'],
                metadata=json.loads(data['metadata']) if data['metadata'] else {}
            )


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, config: SecurityConfig, db_path: Path):
        self.config = config
        self.db_path = db_path
        self._init_database()
        
        # Event buffer for batch processing
        self.event_buffer = deque(maxlen=1000)
        self.buffer_lock = threading.Lock()
        
        # Start background processing
        self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
        self.processing_thread.start()
        
        logger.info("Initialized AuditLogger")
    
    def _init_database(self):
        """Initialize audit database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    event_type TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    action TEXT NOT NULL,
                    result TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    details TEXT,
                    risk_level TEXT DEFAULT 'low'
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_risk_level ON audit_events(risk_level)')
            
            conn.commit()
    
    def log_event(self, event_type: str, resource: str, action: str, result: str,
                 user_id: Optional[str] = None, ip_address: str = "",
                 user_agent: str = "", details: Dict[str, Any] = None,
                 risk_level: str = "low"):
        """Log audit event."""
        event = AuditEvent(
            event_id=secrets.token_urlsafe(16),
            timestamp=datetime.now(),
            user_id=user_id,
            event_type=event_type,
            resource=resource,
            action=action,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
            risk_level=risk_level
        )
        
        with self.buffer_lock:
            self.event_buffer.append(event)
    
    def _process_events(self):
        """Background event processing."""
        while True:
            try:
                events_to_process = []
                
                with self.buffer_lock:
                    while self.event_buffer and len(events_to_process) < 100:
                        events_to_process.append(self.event_buffer.popleft())
                
                if events_to_process:
                    self._save_events(events_to_process)
                
                time.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Error processing audit events: {e}")
                time.sleep(5)
    
    def _save_events(self, events: List[AuditEvent]):
        """Save events to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for event in events:
                cursor.execute('''
                    INSERT INTO audit_events
                    (event_id, timestamp, user_id, event_type, resource, action,
                     result, ip_address, user_agent, details, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id, event.timestamp.isoformat(), event.user_id,
                    event.event_type, event.resource, event.action, event.result,
                    event.ip_address, event.user_agent, json.dumps(event.details),
                    event.risk_level
                ))
            
            conn.commit()
    
    def get_events(self, start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  user_id: Optional[str] = None,
                  event_type: Optional[str] = None,
                  risk_level: Optional[str] = None,
                  limit: int = 1000) -> List[AuditEvent]:
        """Query audit events."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = 'SELECT * FROM audit_events WHERE 1=1'
            params = []
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time.isoformat())
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time.isoformat())
            
            if user_id:
                query += ' AND user_id = ?'
                params.append(user_id)
            
            if event_type:
                query += ' AND event_type = ?'
                params.append(event_type)
            
            if risk_level:
                query += ' AND risk_level = ?'
                params.append(risk_level)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            events = []
            columns = [desc[0] for desc in cursor.description]
            
            for row in rows:
                data = dict(zip(columns, row))
                events.append(AuditEvent(
                    event_id=data['event_id'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    user_id=data['user_id'],
                    event_type=data['event_type'],
                    resource=data['resource'],
                    action=data['action'],
                    result=data['result'],
                    ip_address=data['ip_address'],
                    user_agent=data['user_agent'],
                    details=json.loads(data['details']) if data['details'] else {},
                    risk_level=data['risk_level']
                ))
            
            return events
    
    def cleanup_old_events(self):
        """Clean up old audit events."""
        cutoff_date = datetime.now() - timedelta(days=self.config.audit_log_retention_days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM audit_events WHERE timestamp < ?', 
                          (cutoff_date.isoformat(),))
            deleted_count = cursor.rowcount
            conn.commit()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old audit events")


class SecurityMonitor:
    """Real-time security monitoring and alerting."""
    
    def __init__(self, config: SecurityConfig, audit_logger: AuditLogger):
        self.config = config
        self.audit_logger = audit_logger
        self.alerts = deque(maxlen=1000)
        self.alert_handlers = []
        
        # Monitoring state
        self.suspicious_activities = defaultdict(list)
        self.ip_reputation = defaultdict(int)  # Simple reputation scoring
        
        # Start monitoring
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Initialized SecurityMonitor")
    
    def add_alert_handler(self, handler: Callable[[SecurityAlert], None]):
        """Add alert handler."""
        self.alert_handlers.append(handler)
    
    def create_alert(self, alert_type: str, severity: str, title: str, 
                    description: str, source_ip: Optional[str] = None,
                    affected_user: Optional[str] = None, 
                    details: Dict[str, Any] = None):
        """Create security alert."""
        alert = SecurityAlert(
            alert_id=secrets.token_urlsafe(16),
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            source_ip=source_ip,
            affected_user=affected_user,
            details=details or {}
        )
        
        self.alerts.append(alert)
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"Security alert: {title} ({severity})")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                # Check for suspicious patterns
                self._check_failed_logins()
                self._check_unusual_access_patterns()
                self._check_system_resources()
                self._check_network_connections()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
                time.sleep(60)
    
    def _check_failed_logins(self):
        """Check for suspicious login patterns."""
        # Get recent failed login events
        recent_events = self.audit_logger.get_events(
            start_time=datetime.now() - timedelta(minutes=10),
            event_type='authentication',
            limit=100
        )
        
        failed_logins = [e for e in recent_events if e.result == 'failure']
        
        # Group by IP address
        ip_failures = defaultdict(list)
        for event in failed_logins:
            ip_failures[event.ip_address].append(event)
        
        # Check for brute force attempts
        for ip, failures in ip_failures.items():
            if len(failures) >= 5:  # 5 failures in 10 minutes
                self.create_alert(
                    alert_type='brute_force',
                    severity='high',
                    title='Potential Brute Force Attack',
                    description=f'Multiple failed login attempts from IP {ip}',
                    source_ip=ip,
                    details={'failure_count': len(failures)}
                )
    
    def _check_unusual_access_patterns(self):
        """Check for unusual access patterns."""
        # Get recent access events
        recent_events = self.audit_logger.get_events(
            start_time=datetime.now() - timedelta(hours=1),
            limit=500
        )
        
        # Check for unusual times (outside business hours)
        unusual_time_events = []
        for event in recent_events:
            hour = event.timestamp.hour
            if hour < 6 or hour > 22:  # Outside 6 AM - 10 PM
                unusual_time_events.append(event)
        
        if len(unusual_time_events) > 10:
            self.create_alert(
                alert_type='unusual_access',
                severity='medium',
                title='Unusual Access Time Pattern',
                description='High activity outside normal business hours',
                details={'event_count': len(unusual_time_events)}
            )
    
    def _check_system_resources(self):
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.create_alert(
                    alert_type='resource_usage',
                    severity='warning',
                    title='High CPU Usage',
                    description=f'CPU usage at {cpu_percent}%',
                    details={'cpu_percent': cpu_percent}
                )
            
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.create_alert(
                    alert_type='resource_usage',
                    severity='warning',
                    title='High Memory Usage',
                    description=f'Memory usage at {memory.percent}%',
                    details={'memory_percent': memory.percent}
                )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 90:
                self.create_alert(
                    alert_type='resource_usage',
                    severity='error',
                    title='High Disk Usage',
                    description=f'Disk usage at {disk_percent:.1f}%',
                    details={'disk_percent': disk_percent}
                )
        
        except Exception as e:
            logger.debug(f"System resource check failed: {e}")
    
    def _check_network_connections(self):
        """Check for suspicious network connections."""
        try:
            connections = psutil.net_connections(kind='inet')
            
            # Check for unusual ports
            suspicious_ports = [22, 23, 135, 139, 445, 1433, 3389]  # Common attack targets
            
            for conn in connections:
                if (conn.status == 'ESTABLISHED' and 
                    conn.laddr.port in suspicious_ports and
                    conn.raddr):
                    
                    self.create_alert(
                        alert_type='network_activity',
                        severity='medium',
                        title='Suspicious Network Connection',
                        description=f'Connection on port {conn.laddr.port} from {conn.raddr.ip}',
                        source_ip=conn.raddr.ip,
                        details={
                            'local_port': conn.laddr.port,
                            'remote_ip': conn.raddr.ip,
                            'remote_port': conn.raddr.port
                        }
                    )
        
        except Exception as e:
            logger.debug(f"Network connection check failed: {e}")


class DataProtection:
    """Data protection and encryption utilities."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
        # Initialize encryption
        if config.encryption_key:
            self.cipher = Fernet(config.encryption_key.encode())
        else:
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            logger.warning("Generated new encryption key for data protection")
        
        logger.info("Initialized DataProtection")
    
    def encrypt_data(self, data: Union[str, bytes]) -> bytes:
        """Encrypt sensitive data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.cipher.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data)
    
    def hash_data(self, data: str, salt: Optional[str] = None) -> str:
        """Hash data with optional salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        combined = f"{data}{salt}"
        hash_obj = hashlib.sha256(combined.encode('utf-8'))
        return f"{salt}:{hash_obj.hexdigest()}"
    
    def verify_hash(self, data: str, hashed_data: str) -> bool:
        """Verify hashed data."""
        try:
            salt, hash_value = hashed_data.split(':', 1)
            combined = f"{data}{salt}"
            hash_obj = hashlib.sha256(combined.encode('utf-8'))
            return hash_obj.hexdigest() == hash_value
        except ValueError:
            return False
    
    def classify_data(self, data: Dict[str, Any]) -> str:
        """Classify data sensitivity level."""
        # Simple classification based on field names
        sensitive_fields = ['password', 'ssn', 'credit_card', 'api_key', 'secret']
        confidential_fields = ['email', 'phone', 'address', 'salary']
        
        for field in data.keys():
            field_lower = field.lower()
            if any(sensitive in field_lower for sensitive in sensitive_fields):
                return 'restricted'
            elif any(conf in field_lower for conf in confidential_fields):
                return 'confidential'
        
        return 'internal'
    
    def sanitize_data(self, data: Dict[str, Any], classification: str) -> Dict[str, Any]:
        """Sanitize data based on classification."""
        if classification in ['restricted', 'confidential']:
            sanitized = {}
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 4:
                    # Mask sensitive strings
                    sanitized[key] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                elif isinstance(value, (int, float)):
                    # Mask numbers
                    sanitized[key] = '***'
                else:
                    sanitized[key] = value
            return sanitized
        
        return data


class ComplianceManager:
    """Security compliance and policy enforcement."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.compliance_rules = self._load_compliance_rules()
        
        logger.info("Initialized ComplianceManager")
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules configuration."""
        # Default compliance rules
        return {
            'password_policy': {
                'min_length': self.config.password_min_length,
                'require_special': self.config.password_require_special,
                'max_age_days': 90,
                'history_count': 5
            },
            'access_control': {
                'max_session_duration': self.config.session_timeout_minutes,
                'require_mfa': self.config.enable_2fa,
                'ip_restrictions': self.config.allowed_ip_ranges
            },
            'audit_requirements': {
                'log_retention_days': self.config.audit_log_retention_days,
                'required_events': ['authentication', 'authorization', 'data_access', 'configuration_change']
            },
            'data_protection': {
                'encryption_required': True,
                'classification_levels': self.config.data_classification_levels,
                'retention_policies': {
                    'public': 365,
                    'internal': 2555,  # 7 years
                    'confidential': 2555,
                    'restricted': 3650  # 10 years
                }
            }
        }
    
    def check_compliance(self, area: str) -> Dict[str, Any]:
        """Check compliance for specific area."""
        compliance_results = {
            'area': area,
            'compliant': True,
            'issues': [],
            'recommendations': []
        }
        
        if area == 'password_policy':
            compliance_results.update(self._check_password_compliance())
        elif area == 'access_control':
            compliance_results.update(self._check_access_control_compliance())
        elif area == 'audit_requirements':
            compliance_results.update(self._check_audit_compliance())
        elif area == 'data_protection':
            compliance_results.update(self._check_data_protection_compliance())
        
        return compliance_results
    
    def _check_password_compliance(self) -> Dict[str, Any]:
        """Check password policy compliance."""
        issues = []
        recommendations = []
        
        rules = self.compliance_rules['password_policy']
        
        if self.config.password_min_length < rules['min_length']:
            issues.append(f"Password minimum length ({self.config.password_min_length}) below required ({rules['min_length']})")
        
        if not self.config.password_require_special and rules['require_special']:
            issues.append("Special characters not required in passwords")
            recommendations.append("Enable special character requirement for passwords")
        
        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _check_access_control_compliance(self) -> Dict[str, Any]:
        """Check access control compliance."""
        issues = []
        recommendations = []
        
        rules = self.compliance_rules['access_control']
        
        if self.config.session_timeout_minutes > rules['max_session_duration']:
            issues.append(f"Session timeout ({self.config.session_timeout_minutes}min) exceeds maximum ({rules['max_session_duration']}min)")
        
        if not self.config.enable_2fa and rules['require_mfa']:
            issues.append("Multi-factor authentication not enabled")
            recommendations.append("Enable multi-factor authentication for enhanced security")
        
        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _check_audit_compliance(self) -> Dict[str, Any]:
        """Check audit compliance."""
        issues = []
        recommendations = []
        
        rules = self.compliance_rules['audit_requirements']
        
        if self.config.audit_log_retention_days < rules['log_retention_days']:
            issues.append(f"Audit log retention ({self.config.audit_log_retention_days} days) below required ({rules['log_retention_days']} days)")
        
        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _check_data_protection_compliance(self) -> Dict[str, Any]:
        """Check data protection compliance."""
        issues = []
        recommendations = []
        
        if not self.config.encryption_key:
            issues.append("Encryption key not configured")
            recommendations.append("Configure encryption key for data protection")
        
        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'overall_compliant': True,
            'areas': {}
        }
        
        areas = ['password_policy', 'access_control', 'audit_requirements', 'data_protection']
        
        for area in areas:
            area_result = self.check_compliance(area)
            report['areas'][area] = area_result
            
            if not area_result['compliant']:
                report['overall_compliant'] = False
        
        return report


def create_security_system(config: Dict[str, Any], db_dir: Path) -> Tuple[AuthenticationManager, AuditLogger, SecurityMonitor, DataProtection, ComplianceManager]:
    """
    Factory function to create complete security system.
    
    Args:
        config: Security configuration
        db_dir: Database directory
        
    Returns:
        Tuple of security components
    """
    security_config = SecurityConfig(**config)
    
    db_dir.mkdir(parents=True, exist_ok=True)
    
    auth_manager = AuthenticationManager(security_config, db_dir / 'users.db')
    audit_logger = AuditLogger(security_config, db_dir / 'audit.db')
    security_monitor = SecurityMonitor(security_config, audit_logger)
    data_protection = DataProtection(security_config)
    compliance_manager = ComplianceManager(security_config)
    
    return auth_manager, audit_logger, security_monitor, data_protection, compliance_manager

