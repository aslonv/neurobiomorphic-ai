"""
Production-Grade Monitoring and Logging System

Implements comprehensive monitoring for neurobiomorphic AI systems:
- Performance metrics tracking
- Memory and computational resource monitoring  
- Model behavior analysis
- Distributed logging with structured formats
- Real-time alerting and anomaly detection
- Integration with popular monitoring tools (W&B, TensorBoard, MLflow)
"""

import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import sys
import os
from pathlib import Path
import traceback
from collections import defaultdict, deque
import numpy as np
import torch


@dataclass
class MetricSnapshot:
    """Snapshot of system metrics at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_memory_used_gb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    inference_time: Optional[float] = None
    training_time: Optional[float] = None
    throughput: Optional[float] = None  # samples per second
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class StructuredLogger:
    """
    Structured logging with JSON format for production environments.
    
    Provides consistent log formatting across all system components
    with contextual information and severity levels.
    """
    
    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        enable_console: bool = True,
        enable_file: bool = True,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        self.logger.handlers.clear()
        
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        json_formatter = JsonFormatter()
        
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(level)
            self.logger.addHandler(console_handler)
        
        if enable_file and log_file:
            from logging.handlers import RotatingFileHandler
            
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setFormatter(json_formatter)
            file_handler.setLevel(level)
            self.logger.addHandler(file_handler)
        
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set persistent context for all log messages."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear persistent context."""
        self.context.clear()
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Internal method to add context to log messages."""
        full_context = {**self.context, **kwargs}
        
        extra = {
            'context': full_context,
            'timestamp': datetime.utcnow().isoformat(),
            'component': self.name
        }
        
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with context and traceback."""
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
            kwargs['traceback'] = traceback.format_exc()
        
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add context if available
        if hasattr(record, 'context'):
            log_entry['context'] = record.context
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class SystemMonitor:
    """
    System resource monitoring with real-time metrics collection.
    
    Monitors CPU, memory, GPU usage and provides alerts for anomalies.
    """
    
    def __init__(
        self,
        collection_interval: float = 1.0,
        history_size: int = 1000,
        enable_gpu_monitoring: bool = True
    ):
        self.collection_interval = collection_interval
        self.history_size = history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        
        # Metric storage
        self.metrics_history = deque(maxlen=history_size)
        self.current_metrics = None
        
        # Monitoring thread
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Alert thresholds
        self.cpu_threshold = 90.0  # CPU usage %
        self.memory_threshold = 90.0  # Memory usage %
        self.gpu_memory_threshold = 90.0  # GPU memory %
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Logger
        self.logger = StructuredLogger("SystemMonitor")
        
    def add_alert_callback(self, callback: Callable[[str, MetricSnapshot], None]):
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if self._monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("Started system monitoring")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped system monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=e)
                time.sleep(self.collection_interval)
    
    def _collect_metrics(self) -> MetricSnapshot:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024 ** 3)
        
        # GPU metrics
        gpu_memory_used_gb = None
        gpu_utilization = None
        
        if self.enable_gpu_monitoring:
            try:
                # Using torch for GPU memory
                gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                
                # Try to get GPU utilization (requires nvidia-ml-py or similar)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = gpu_util.gpu
                except ImportError:
                    pass  # GPU utilization not available
                    
            except Exception:
                pass  # GPU metrics not available
        
        return MetricSnapshot(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_utilization=gpu_utilization
        )
    
    def _check_alerts(self, metrics: MetricSnapshot):
        """Check metrics against thresholds and trigger alerts."""
        alerts = []
        
        if metrics.cpu_percent > self.cpu_threshold:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.memory_threshold:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if (metrics.gpu_memory_used_gb is not None and 
            torch.cuda.is_available()):
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_memory_percent = (metrics.gpu_memory_used_gb / gpu_memory_total) * 100
            
            if gpu_memory_percent > self.gpu_memory_threshold:
                alerts.append(f"High GPU memory usage: {gpu_memory_percent:.1f}%")
        
        # Trigger alert callbacks
        for alert_message in alerts:
            self.logger.warning(alert_message, 
                              cpu_percent=metrics.cpu_percent,
                              memory_percent=metrics.memory_percent,
                              gpu_memory_gb=metrics.gpu_memory_used_gb)
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert_message, metrics)
                except Exception as e:
                    self.logger.error("Error in alert callback", error=e)
    
    def get_current_metrics(self) -> Optional[MetricSnapshot]:
        """Get current system metrics."""
        return self.current_metrics
    
    def get_metrics_history(self, duration: Optional[timedelta] = None) -> List[MetricSnapshot]:
        """Get metrics history within specified duration."""
        if duration is None:
            return list(self.metrics_history)
        
        cutoff_time = datetime.utcnow() - duration
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def get_average_metrics(self, duration: Optional[timedelta] = None) -> Dict[str, float]:
        """Get average metrics over specified duration."""
        history = self.get_metrics_history(duration)
        
        if not history:
            return {}
        
        metrics = {
            'cpu_percent': np.mean([m.cpu_percent for m in history]),
            'memory_percent': np.mean([m.memory_percent for m in history]),
            'memory_used_gb': np.mean([m.memory_used_gb for m in history])
        }
        
        # GPU metrics if available
        gpu_memory_values = [m.gpu_memory_used_gb for m in history if m.gpu_memory_used_gb is not None]
        gpu_util_values = [m.gpu_utilization for m in history if m.gpu_utilization is not None]
        
        if gpu_memory_values:
            metrics['gpu_memory_used_gb'] = np.mean(gpu_memory_values)
        
        if gpu_util_values:
            metrics['gpu_utilization'] = np.mean(gpu_util_values)
        
        return metrics


class PerformanceProfiler:
    """
    Performance profiling for machine learning operations.
    
    Tracks training time, inference time, memory usage, and custom metrics.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.logger = StructuredLogger(f"Profiler.{name}")
        
        # Timing storage
        self.timings = defaultdict(list)
        self.active_timers = {}
        
        # Memory tracking
        self.memory_snapshots = []
        
        # Custom metrics
        self.custom_metrics = defaultdict(list)
        
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.active_timers[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.active_timers:
            self.logger.warning(f"Timer '{operation}' was not started")
            return 0.0
        
        start_time = self.active_timers.pop(operation)
        duration = time.time() - start_time
        self.timings[operation].append(duration)
        
        self.logger.debug(f"Operation '{operation}' took {duration:.4f} seconds")
        return duration
    
    def time_operation(self, operation: str):
        """Context manager for timing operations."""
        return TimingContext(self, operation)
    
    def record_memory_snapshot(self, label: str = ""):
        """Record current memory usage."""
        snapshot = {
            'label': label,
            'timestamp': datetime.utcnow(),
            'cpu_memory_gb': psutil.virtual_memory().used / (1024 ** 3),
        }
        
        if torch.cuda.is_available():
            snapshot['gpu_memory_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
            snapshot['gpu_memory_cached_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)
        
        self.memory_snapshots.append(snapshot)
        return snapshot
    
    def record_custom_metric(self, name: str, value: float):
        """Record a custom performance metric."""
        self.custom_metrics[name].append({
            'value': value,
            'timestamp': datetime.utcnow()
        })
    
    def get_timing_stats(self, operation: str) -> Dict[str, float]:
        """Get statistical summary of operation timings."""
        if operation not in self.timings or not self.timings[operation]:
            return {}
        
        timings = np.array(self.timings[operation])
        
        return {
            'count': len(timings),
            'mean': float(np.mean(timings)),
            'std': float(np.std(timings)),
            'min': float(np.min(timings)),
            'max': float(np.max(timings)),
            'median': float(np.median(timings)),
            'p95': float(np.percentile(timings, 95)),
            'p99': float(np.percentile(timings, 99))
        }
    
    def get_all_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all operations."""
        return {op: self.get_timing_stats(op) for op in self.timings.keys()}
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'profiler_name': self.name,
            'report_timestamp': datetime.utcnow().isoformat(),
            'timing_stats': self.get_all_timing_stats(),
            'memory_snapshots': self.memory_snapshots[-10:],  # Last 10 snapshots
            'custom_metrics': {}
        }
        
        # Summarize custom metrics
        for name, values in self.custom_metrics.items():
            if values:
                metric_values = [v['value'] for v in values]
                report['custom_metrics'][name] = {
                    'count': len(metric_values),
                    'mean': float(np.mean(metric_values)),
                    'std': float(np.std(metric_values)),
                    'min': float(np.min(metric_values)),
                    'max': float(np.max(metric_values))
                }
        
        return report
    
    def save_report(self, filepath: str):
        """Save performance report to file."""
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Saved performance report to {filepath}")
    
    def clear_history(self):
        """Clear all collected performance data."""
        self.timings.clear()
        self.memory_snapshots.clear()
        self.custom_metrics.clear()
        self.logger.info("Cleared performance history")


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, profiler: PerformanceProfiler, operation: str):
        self.profiler = profiler
        self.operation = operation
    
    def __enter__(self):
        self.profiler.start_timer(self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_timer(self.operation)


class ExperimentTracker:
    """
    Comprehensive experiment tracking with integration to popular tools.
    
    Supports W&B, TensorBoard, MLflow, and custom logging backends.
    """
    
    def __init__(
        self,
        experiment_name: str,
        project_name: str = "neurobiomorphic-ai",
        backends: List[str] = ["tensorboard"],
        config: Optional[Dict[str, Any]] = None
    ):
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.backends = backends
        self.config = config or {}
        
        self.logger = StructuredLogger(f"Tracker.{experiment_name}")
        
        # Initialize backends
        self.backend_instances = {}
        self._initialize_backends()
        
        # Metric storage
        self.metrics = defaultdict(list)
        self.step_counter = 0
        
    def _initialize_backends(self):
        """Initialize tracking backends."""
        for backend in self.backends:
            try:
                if backend == "wandb":
                    self._init_wandb()
                elif backend == "tensorboard":
                    self._init_tensorboard()
                elif backend == "mlflow":
                    self._init_mlflow()
                else:
                    self.logger.warning(f"Unknown backend: {backend}")
            except Exception as e:
                self.logger.error(f"Failed to initialize {backend}", error=e)
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            import wandb
            
            wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=self.config
            )
            
            self.backend_instances['wandb'] = wandb
            self.logger.info("Initialized W&B tracking")
            
        except ImportError:
            self.logger.warning("W&B not available, install with: pip install wandb")
    
    def _init_tensorboard(self):
        """Initialize TensorBoard tracking."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            log_dir = f"runs/{self.project_name}/{self.experiment_name}"
            writer = SummaryWriter(log_dir=log_dir)
            
            self.backend_instances['tensorboard'] = writer
            self.logger.info(f"Initialized TensorBoard logging to {log_dir}")
            
        except ImportError:
            self.logger.warning("TensorBoard not available, install with: pip install tensorboard")
    
    def _init_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            import mlflow
            
            mlflow.set_experiment(self.project_name)
            mlflow.start_run(run_name=self.experiment_name)
            
            # Log config parameters
            for key, value in self.config.items():
                mlflow.log_param(key, value)
            
            self.backend_instances['mlflow'] = mlflow
            self.logger.info("Initialized MLflow tracking")
            
        except ImportError:
            self.logger.warning("MLflow not available, install with: pip install mlflow")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all configured backends."""
        if step is None:
            step = self.step_counter
            self.step_counter += 1
        
        # Store metrics locally
        for name, value in metrics.items():
            self.metrics[name].append({'value': value, 'step': step})
        
        # Log to backends
        for backend_name, backend in self.backend_instances.items():
            try:
                if backend_name == "wandb":
                    backend.log(metrics, step=step)
                elif backend_name == "tensorboard":
                    for name, value in metrics.items():
                        backend.add_scalar(name, value, step)
                elif backend_name == "mlflow":
                    for name, value in metrics.items():
                        backend.log_metric(name, value, step)
            except Exception as e:
                self.logger.error(f"Error logging to {backend_name}", error=e)
        
        # Log summary
        metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metric_str}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log hyperparameters and configuration."""
        self.config.update(params)
        
        for backend_name, backend in self.backend_instances.items():
            try:
                if backend_name == "wandb":
                    backend.config.update(params)
                elif backend_name == "mlflow":
                    for key, value in params.items():
                        backend.log_param(key, value)
                # TensorBoard doesn't have direct parameter logging
            except Exception as e:
                self.logger.error(f"Error logging parameters to {backend_name}", error=e)
        
        self.logger.info(f"Logged parameters: {params}")
    
    def log_model_artifact(self, model_path: str, artifact_name: str = "model"):
        """Log model artifacts."""
        for backend_name, backend in self.backend_instances.items():
            try:
                if backend_name == "wandb":
                    backend.save(model_path)
                elif backend_name == "mlflow":
                    backend.log_artifact(model_path, artifact_name)
            except Exception as e:
                self.logger.error(f"Error logging artifact to {backend_name}", error=e)
        
        self.logger.info(f"Logged model artifact: {model_path}")
    
    def finish(self):
        """Finish experiment tracking."""
        for backend_name, backend in self.backend_instances.items():
            try:
                if backend_name == "wandb":
                    backend.finish()
                elif backend_name == "tensorboard":
                    backend.close()
                elif backend_name == "mlflow":
                    backend.end_run()
            except Exception as e:
                self.logger.error(f"Error finishing {backend_name}", error=e)
        
        self.logger.info("Finished experiment tracking")


class AlertManager:
    """
    Centralized alert management for production systems.
    
    Handles notifications via multiple channels (email, Slack, webhooks).
    """
    
    def __init__(self):
        self.logger = StructuredLogger("AlertManager")
        self.alert_handlers = []
        
    def add_handler(self, handler: Callable[[str, Dict[str, Any]], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
    
    def send_alert(self, message: str, severity: str = "warning", **context):
        """Send alert through all configured handlers."""
        alert_data = {
            'message': message,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat(),
            'context': context
        }
        
        self.logger.warning(f"ALERT [{severity.upper()}]: {message}", **context)
        
        for handler in self.alert_handlers:
            try:
                handler(message, alert_data)
            except Exception as e:
                self.logger.error("Error in alert handler", error=e)


# Global instances for easy access
global_logger = StructuredLogger("neurobiomorphic")
global_monitor = SystemMonitor()
global_profiler = PerformanceProfiler("global")
global_alert_manager = AlertManager()


def setup_monitoring(
    log_level: str = "INFO",
    log_file: Optional[str] = "logs/neurobiomorphic.log",
    enable_system_monitoring: bool = True,
    monitoring_interval: float = 1.0
):
    """Setup global monitoring and logging."""
    # Configure global logger
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    global_logger.logger.setLevel(level_map.get(log_level, logging.INFO))
    
    # Start system monitoring
    if enable_system_monitoring:
        global_monitor.collection_interval = monitoring_interval
        global_monitor.start_monitoring()
    
    global_logger.info("Monitoring system initialized",
                      log_level=log_level,
                      system_monitoring=enable_system_monitoring,
                      monitoring_interval=monitoring_interval)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for a component."""
    return StructuredLogger(name)